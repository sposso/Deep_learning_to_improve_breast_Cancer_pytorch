'''This script contains the helper functions to use 
the patch classifier in a sliding window fashion and 
get the heatmaps (saliency maps)'''

import numpy as np 
import torch
from torchvision import models
import torch.nn as nn



def add_img_margins(img, margin_size):
    '''Add all zero margins to an image
    '''
    enlarged_img = np.zeros((img.shape[0]+margin_size*2, 
                             img.shape[1]+margin_size*2))
    enlarged_img[margin_size:margin_size+img.shape[0], 
                 margin_size:margin_size+img.shape[1]] = img
    return enlarged_img

def sweep_img_patches(img, patch_size, stride):
    '''Generate the patches through whole image'''

    nb_row = round(float(img.shape[0] - patch_size)/stride + .49)
    nb_col = round(float(img.shape[1] - patch_size)/stride + .49)
    nb_row = int(nb_row)
    nb_col = int(nb_col)
    sweep_hei = patch_size + (nb_row - 1)*stride
    sweep_wid = patch_size + (nb_col - 1)*stride
    y_gap = int((img.shape[0] - sweep_hei)/2)
    x_gap = int((img.shape[1] - sweep_wid)/2)
    patch_list = []
    for y in range(y_gap, y_gap + nb_row*stride, stride):
        for x in range(x_gap, x_gap + nb_col*stride, stride):
            patch = img[y:y+patch_size, x:x+patch_size].copy()
            patch_list.append(patch.astype('float32'))
    return np.stack(patch_list), nb_row, nb_col


def initialize_model(model_name, num_classes,use_pretrained):
    '''The architecture employed for the patch classifier is the 
    resnet50. Since this patch model classifies 5 patch categories, it is 
    necessary to adjust the default  resnet50 of pytorch, which classifies 
    1000 classes'''
    model_ft = None
    

    if model_name == "resnet":
        """ Resnet50
        """
        model_ft = models.resnet50()
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        if use_pretrained =='s':
            #Load the trained patch classifier's parameters on the s dataset. 
            model_ft.load_state_dict(torch.load('.../best_s_patch_model.pt',map_location=torch.device('cpu')))
            
            print('Initialzing model with s Patch Classifier Weights')
            #Load the trained patch classifier's parameters on the s10 dataset.
        elif use_pretrained == 's10':
            model_ft.load_state_dict(torch.load('.../best_s10_patch_model.pt',map_location=torch.device('cpu')))
            print('Initialzing model with s10 Patch Classifier Weights')
            
            
        
    return model_ft
