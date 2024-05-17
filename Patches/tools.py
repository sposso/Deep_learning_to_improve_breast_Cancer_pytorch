import numpy as np
import cv2
import pandas as pd
from imageio import imread
import matplotlib.pyplot as plt
import png
from scipy import ndimage
import sys

def image_as_png(image, png_filename, bitdepth=16):

    
    with open(png_filename, 'wb') as f:
        writer = png.Writer(
            height=image.shape[0],
            width=image.shape[1],
            bitdepth=bitdepth,
            greyscale=True
        )
        writer.write(f, image.tolist())
        
        print("done!")
        
        
def crop_val(v, minv, maxv):
    v = v if v >= minv else minv
    v = v if v <= maxv else maxv
    return v

def add_img_margins(img, margin_size):
    '''Add all zero margins to an image
    '''
    enlarged_img = np.zeros((img.shape[0]+margin_size*2, 
                             img.shape[1]+margin_size*2))
    enlarged_img[margin_size:margin_size+img.shape[0], 
                 margin_size:margin_size+img.shape[1]] = img
    return enlarged_img
  
def segment_breast( img,mask, low_int_threshold=.05, crop=True):
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
        img_mask = cv2.bitwise_and(mask, mask, mask=breast_mask)
        x,y,w,h = cv2.boundingRect(contours[idx])
        if crop:
            img_breast_only = img_breast_only[y:y+h, x:x+w]
            mask = img_mask[y:y+h, x:x+w]
            
        return img_breast_only, mask
      
def overlap_patch_roi(patch_center, patch_size, roi_mask, 
                      add_val=1000, cutoff=.9):
    
    '''This function returns true if the patch satisfies
    the minimum overlapping area with the region of interest'''
    x1,y1 = (patch_center[0] - int(patch_size/2), 
             patch_center[1] - int(patch_size/2))
    x2,y2 = (patch_center[0] + int(patch_size/2), 
             patch_center[1] + int(patch_size/2))
    x1 = crop_val(x1, 0, roi_mask.shape[1])
    y1 = crop_val(y1, 0, roi_mask.shape[0])
    x2 = crop_val(x2, 0, roi_mask.shape[1])
    y2 = crop_val(y2, 0, roi_mask.shape[0])
    
    roi_area = (roi_mask>0).sum()
    roi_patch_added = roi_mask.copy()
    roi_patch_added = roi_patch_added.astype('uint16')
    roi_patch_added[y1:y2, x1:x2] += add_val
    patch_area = (roi_patch_added>=add_val).sum()
    inter_area = (roi_patch_added>add_val).sum().astype('float32')
   
    return (inter_area/roi_area > cutoff or inter_area/patch_area > cutoff)
