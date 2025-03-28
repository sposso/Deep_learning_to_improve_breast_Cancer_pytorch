
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import pandas as pd
import numpy as np
from torch import nn
from PIL import Image
import cv2
import random


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
        imgs = torch.cat([img_t for img_t,_ in data],dim=0)
        imgs_mean = imgs.reshape(-1).mean()
        imgs_std = imgs.reshape(-1).std

    elif channels == 3:
        imgs = torch.stack([img_t for img_t,_ in data],dim=3)

        imgs_mean = imgs.reshape(3,-1).mean(dim=1)

        imgs_std = imgs.reshape(3,-1).std(dim=1)

    return imgs_mean, imgs_std


def segment_breast( img, low_int_threshold=.05, crop=False):
    '''Perform breast segmentation
    Args:
        low_int_threshold([float or int]): Low intensity threshold to 
                filter out background. It can be a fraction of the max 
                intensity value or an integer intensity value.
        crop ([bool]): Whether or not to crop the image.
    Returns:
        An image of the segmented breast.
    NOTES: the low_int_threshold is applied to an image of dtype 'uint8',
        which has a max value of 255.
    '''
    # Create img for thresholding and contours.
    img_8u = (img.astype('float32')/img.max()*255).astype('uint8')
    if low_int_threshold < 1.:
        low_th = int(img_8u.max()*low_int_threshold)
    else:
        low_th = int(low_int_threshold)
    _, img_bin = cv2.threshold(
        img_8u, low_th, maxval=255, type=cv2.THRESH_BINARY)
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        contours,_ = cv2.findContours(
            img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours,_ = cv2.findContours(
            img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_areas = [ cv2.contourArea(cont) for cont in contours ]
    idx = np.argmax(cont_areas)  # find the largest contour, i.e. breast.
    breast_mask = cv2.drawContours(
        np.zeros_like(img_bin), contours, idx, 255, -1)  # fill the contour.
    # segment the breast.
    img_breast_only = cv2.bitwise_and(img, img, mask=breast_mask)
    x,y,w,h = cv2.boundingRect(contours[idx])
    if crop:
        img_breast_only = img_breast_only[y:y+h, x:x+w]

            
    return img_breast_only

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

            image = pil_loader(img_path)*(1/65535)
            image = segment_breast(image)
        elif extension =='npy':
            image = np.float32(np.load(img_path)*(1/65535))
            image = segment_breast(image)
            
        y_label = torch.tensor(int(self.annotations.iloc[index,1]))

        if self.transform is not None:
            image = self.transform(image)

        return image, y_label
