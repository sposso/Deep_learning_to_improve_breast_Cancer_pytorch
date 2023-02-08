
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import torchvision.transforms as T
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import time
import copy
from PIL import Image

import random

def pil_loader(path: str) :
    with open(path, "rb") as f:
        img = Image.open(f)
        img = np.float32(np.asarray(img))
        return img

class CBIS_MAMMOGRAM(Dataset):
    def __init__(self, csv_file, transform=None):

        self.annotations = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = self.annotations.iloc[index,0]
        extension = img_path.split(".")[-1]
        if extension =='png':

            image = pil_loader(img_path)
        elif extension =='npy':
            image = np.float32(np.load(img_path)*(1/65535))
            
        y_label = torch.tensor(int(self.annotations.iloc[index,1]))

        if self.transform is not None:
            image = self.transform(image)

        return image, y_label

class MySubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

class MyIntensityShift:
    """Intensity shift ± 20 %."""

    def __init__(self, shift):
        self.shift = shift

    def __call__(self, x):
        f = random.randint(self.shift[0],self.shift[1])/100
        return f*x

def mean_std(train, channels = 1):
    data = CBIS_MAMMOGRAM(train, transforms = T.ToTensor())
    if channels == 1:
        imgs = torch.cat([img_t for img_t,_ in train],dim=0)
        mean = imgs.reshape(-1).mean()
        std = imgs.reshape(-1).std

    elif channels == 3:
        imgs = torch.stack([img_t for img_t,_ in train],dim=3)

        imgs_mean = imgs.reshape(3,-1).mean(dim=1)

        imgs_std = imgs.reshape(3,-1).std(dim=1)
