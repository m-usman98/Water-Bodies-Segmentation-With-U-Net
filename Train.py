# python -m visdom.server

import os
import time
from glob import glob
import torch as py
from torch.utils.data import DataLoader
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from Data import LoadData
from Model import Unet
from torchvision import transforms
import numpy as np
from visdom import Visdom


# Hyperparameters

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
BATCH_SIZE = 4
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
SAVE_MODEL = "Model.pth"
NUM_WORKERS = 2
DEVICE = "cuda" if py.cuda.is_available() else "cpu"



# Graph Plotter Class
class GraphPlotter(object):
    # Initializes the initial values
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    # A Function to Plot the Graph
    def plot(self, Var_Name, Split_Name, Title_Name, x, y):
        # Plots the graph If the Var_Name is not Null
        if Var_Name not in self.plots:
            self.plots[Var_Name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env, opts=dict(
                legend=[Split_Name],
                title=Title_Name,
                xlabel='Epochs',
                ylabel=Var_Name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[Var_Name], name=Split_Name,
                          update='append')

plotter = GraphPlotter(env_name='main')
plotter1 = GraphPlotter(env_name='main')

def Train(loader, model, optimizer, loss_fn, device):
    epoch_loss = 0.0

    model.train()
    for x, y in loader:
        x = x.to(device, dtype=py.float32)
        y = y.to(device, dtype=py.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(loader)
    return epoch_loss


def Eval(loader, model, loss_fn, device):
    epoch_loss = 0.0

    model.eval()
    with py.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=py.float32)
            y = y.to(device, dtype=py.float32)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(loader)
    return epoch_loss


if __name__ == "__main__":

    # Load dataset
    train_x = glob("./Dataset/Train/Image/*")
    train_y = glob("./Dataset/Train/Mask/*")

    valid_x = glob("./Dataset/Val/Image/*")
    valid_y = glob("./Dataset/Val/Mask/*")

    data_len = f"Dataset Size:\nTraining Dataset: {len(train_x)} - Validation Dataset: {len(valid_x)}\n"
    print(data_len)

    transform = transforms.Compose([transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
                                    transforms.RandomRotation(10),
                                    transforms.RandomHorizontalFlip(0.2)])

    # Dataset and loader
    train_dataset = LoadData(train_x, train_y, transform)
    valid_dataset = LoadData(valid_x, valid_y, transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = Unet().to(DEVICE)

    optimizer = py.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = py.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = nn.BCEWithLogitsLoss()

    # Training the model
    best_valid_loss = float("inf")

    for epoch in range(NUM_EPOCHS):

        train_loss = Train(train_loader, model, optimizer, loss_fn, DEVICE)
        valid_loss = Eval(valid_loader, model, loss_fn, DEVICE)

        # Saving the model
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving Model: {SAVE_MODEL}"
            print(data_str)

            best_valid_loss = valid_loss
            py.save(model.state_dict(), SAVE_MODEL)

        data_str = f'Epoch: {epoch + 1:02}\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        plotter.plot('Training', 'Training', 'Training Loss', epoch + 1, train_loss)
        plotter1.plot('Validation Testing', 'Validation', 'Validation Loss', epoch + 1, valid_loss)
        print(data_str)
