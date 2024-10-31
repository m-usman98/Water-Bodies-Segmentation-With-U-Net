import time
from operator import add
import numpy as np
from glob import glob
import cv2 as cv
from tqdm import tqdm
import os
import imageio
import torch as py
from sklearn.metrics import accuracy_score, jaccard_score, precision_score
from Model import Unet

Load_MODEL = "Model.pth"
DEVICE = "cuda" if py.cuda.is_available() else "cpu"


def calculate_metrics(y_true, y_pred):
    # Ground truth
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    # Prediction
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_precision, score_acc]


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask

if __name__ == "__main__":

    # Load Test Dataset - Unseen
  # test_x = glob("./Dataset/Test-Unseen/image/*")
  # test_y = glob("./Dataset/Test-Unseen/mask/*")

    # Load Test Dataset - Seen
    test_x = glob("./Dataset/Test-Seen/image/*")
    test_y = glob("./Dataset/Test-Seen/mask/*")

    model = Unet().to(DEVICE)
    model.load_state_dict(py.load(Load_MODEL, map_location=DEVICE))
    model.eval()

    metrics_score = [0.0, 0.0, 0.0]
    time_taken = []
    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):


        # Extract the name
        name = x.split("/")[-1].split(".")[0]
        Delete_Iest_Images = x.split("/")[-1]
        Delete_Iest_Mask_Images = y.split("/")[-1]


        # Reading image
        image = cv.imread(x, cv.IMREAD_COLOR)
        image = cv.resize(image, (512, 512))
        x = np.transpose(image, (2, 0, 1))
        x = x / 255.0
        x = np.expand_dims(x, axis=0)
        x = x.astype(np.float32)
        x = py.from_numpy(x)
        x = x.to(DEVICE)

        # Reading mask
        mask = cv.imread(y, cv.IMREAD_GRAYSCALE)
        mask = cv.resize(mask, (512, 512))
        y = np.expand_dims(mask, axis=0)
        y = y / 255.0
        y = np.expand_dims(y, axis=0)
        y = y.astype(np.float32)
        y = py.from_numpy(y)
        y = y.to(DEVICE)

        with py.no_grad():
            # Prediction and Calculating FPS
            start_time = time.time()
            pred_y = model(x)
            pred_y = py.sigmoid(pred_y)
            total_time = time.time() - start_time
            time_taken.append(total_time)

            score = calculate_metrics(y, pred_y)
            metrics_score = list(map(add, metrics_score, score))
            pred_y = pred_y[0].cpu().numpy()
            pred_y = np.squeeze(pred_y, axis=0)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

        ori_mask = mask_parse(mask)
        pred_y = mask_parse(pred_y)
        line = np.ones(((512, 512)[1], 10, 3)) * 128

        # Saves the Final Output along with mask and Original Image
        cat_images = np.concatenate(
            [image, line, ori_mask, line, pred_y * 255], axis=1
        )
        cv.imwrite("./Output/{}.jpg".format(i), cat_images)

    jaccard = metrics_score[0] / len(test_x)
    precision = float(metrics_score[1]) / len(test_x)
    acc = float(metrics_score[2]) / len(test_x)

    print(
            f"Jaccard: {jaccard:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f}")





