import cv2 as cv
import numpy as np
import torch as py
from torch.utils.data import Dataset


class LoadData(Dataset):
    def __init__(self, image_pth, mask_pth, transform):
        self.image_pth = image_pth
        self.mask_pth = mask_pth
        self.n_samples = len(image_pth)
        self.transform = transform

    def __getitem__(self, index):
        # Reading image
        image = cv.imread(self.image_pth[index], cv.IMREAD_COLOR)
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = py.from_numpy(image)

        # Read Mask
        mask = cv.imread(self.mask_pth[index], cv.IMREAD_GRAYSCALE)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = py.from_numpy(mask)

        return self.transform(image), self.transform(mask)

    def __len__(self):
        return self.n_samples
