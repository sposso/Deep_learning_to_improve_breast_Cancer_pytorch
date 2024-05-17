import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np 
import torch.optim as optim
import torch.nn as nn
import datetime
from sklearn.model_selection import train_test_split
import time
import copy
from imageio import imread
import random

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


def split_unbalanced_data(dataframe_path):
  ''' This function split the dataset into train set, validation set and test set. Since CBIS MAMMOGRAM contains
  unbalanced classes, the split must keep the same proportion of cases in each set.'''
    df = pd.read_csv(path)
    target = df.label.to_numpy()
    train_indices, test_indices = train_test_split(np.arange(target.shape[0]), test_size= 0.15, train_size=0.85, stratify=target, random_state= 42)

    train_dataset = Subset(mammograms, indices=train_indices)
    test_dataset = Subset(mammograms, indices=test_indices)
    df_train = df.loc[train_indices,:]
    df_train = df_train.reset_index(drop = True )
    label_train = df_train.label.to_numpy()
    train_in, validation_in = train_test_split(np.arange(train_indices.shape[0]), test_size =0.1, train_size = 0.9, stratify =label_train, random_state= 42)

    #subsets 
    train_set = Subset(train_dataset, indices=train_in)
    validation_dataset = Subset(train_dataset, indices=validation_in)
    
    return train_set,validation_set,test_dataset

class CBIS_MAMMOGRAM(Dataset):
  ''' The path of each patch image with its respective label 
  is saved inn a excel spreedsheet'''
    def __init__(self, csv_file, transform=None):
        
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = self.annotations.iloc[index,0]
        image = imread(img_path)*(1/65535)    
        image = np.stack([image, image, image])
        #Shape expected by pytorch models
        image = np.moveaxis(image, 0, -1)
        y_label = torch.tensor(int(self.annotations.iloc[index,1]))

        if self.transform is not None:
            image = self.transform(image)

        return image, y_label
   
def initialize_model(model_name, num_classes,use_pretrained=True):
  '''The architecture employed for the patch classifier is 
    resnet50. Since this patch model classifies 5 patch categories, it is 
    necessary to adjust the default  resnet50 of pytorch, which classifies 
    1000 classes'''
  model_ft = None


  if model_name == "resnet":
      """ Resnet50
      """
      model_ft = models.resnet50(pretrained=use_pretrained)
      num_ftrs = model_ft.fc.in_features
      model_ft.fc = nn.Linear(num_ftrs, num_classes)

  return model_ft

