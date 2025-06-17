import numpy as np
import cv2
import pandas as pd
from imageio import imread
import png
from scipy import ndimage
import sys
import os 


def image_as_png(image, png_filename, bitdepth=16):
    '''Save images in PNG format.

    Medical images usually are 16-bit images.
    '''
    
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
    '''This function guarantees that patches are 
    within the image'''
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


def sample_patches(img_,index,label, roi_mask_,folder_s, folder_s10,df_s,df_s10,patch_size=224,
                   pos_cutoff=.9, neg_cutoff=.35,
                   nb_bkg=11, nb_abn=10, start_sample_nb=0,
                   verbose=True):
    
    '''This function generates the s and the s10 patch image dataset.
        -S dataset corresponds to the sets of patches in which one is centered
        on the region of interest (ROI) and one is a random background patch from 
        the same image
        -S10 datasets is derived from 10 sampled patches around each ROi with a minimum 
        overlapping ratio (pos_cutoff) of 0.9 and the same number of background patches from 
        the same image'''

    index = int(index)
    print(index)
    img,roi_mask = segment_breast(img_,roi_mask_)
    if roi_mask.sum()== 0:
        roi_mask = roi_mask_
        img= img_
    
    #Check if the mask and the image sizes are equal.
    if img.shape == roi_mask.shape:
        
        print("img and mask shape match")
        
        img = add_img_margins(img, int(patch_size/2))
        roi_mask = add_img_margins(roi_mask, int(patch_size/2))
        roi_mask = roi_mask.astype("uint8")
       
        # Get ROI bounding box.
        _, thresh = cv2.threshold(roi_mask, 254, 255, cv2.THRESH_BINARY)


        contours,_ = cv2.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cont_areas = [ cv2.contourArea(cont) for cont in contours ]
        
        idx = np.argmax(cont_areas)  # find the largest contour.
        rx,ry,rw,rh = cv2.boundingRect(contours[idx])

        cy,cx = ndimage.measurements.center_of_mass(thresh)
        cy = int(round(cy,0))
        cx = int(round(cx,0))

        print("ROI centroid=", (cx,cy)); sys.stdout.flush()
        patch_center = (cx,cy)
        x1,y1 = (patch_center[0] - int(patch_size/2), 
                 patch_center[1] - int(patch_size/2))
        x2,y2 = (patch_center[0] + int(patch_size/2), 
                 patch_center[1] + int(patch_size/2))
        x1 = crop_val(x1, 0, roi_mask.shape[1])
        y1 = crop_val(y1, 0, roi_mask.shape[0])
        x2 = crop_val(x2, 0, roi_mask.shape[1])
        y2 = crop_val(y2, 0, roi_mask.shape[0])

        #generating and saving a patch for the S set 
        s_patch = img[y1:y2, x1:x2]
        s_patch= s_patch.astype('uint16')
        
        path_s = os.path.join(folder_s,f"roi_{index}.png")
        image_as_png(s_patch,path_s)
        
        df_s.loc[index] = [path_s, label]


        rng = np.random.RandomState(321)
        # Sample abnormality first.
        sampled_abn = 0
        nb_try = 0
        
        while sampled_abn < nb_abn:
            x = rng.randint(rx, rx + rw)
            y = rng.randint(ry, ry + rh)
            nb_try += 1
            if nb_try > 1000:
                print("Nb of trials reached maximum, decrease overlap cutoff by 0.05")
                sys.stdout.flush()
                pos_cutoff -= .05
                nb_try = 0
                if pos_cutoff <= .0:
                    raise Exception("overlap cutoff becomes non-positive, "
                                    "check roi mask input.")
           

            if overlap_patch_roi((x,y), patch_size, roi_mask, cutoff=pos_cutoff):
                patch = img[y - int(patch_size/2):y + int(patch_size/2), 
                            x - int(patch_size/2):x + int(patch_size/2)]

                path_s10 = os.path.join(folder_s10,f"roi_{index}_{sampled_abn}.png")
                
                patch= patch.astype('uint16')
                
                image_as_png(patch,path_s10)
                
                df_s10.loc[index] = [path_s10, label] 
               
                sampled_abn += 1
                nb_try = 0
                if verbose:
                    print("sampled an abn patch at (x,y) center=", (x,y))
                    sys.stdout.flush()
        
        # Sample background.
        
        sampled_bkg = start_sample_nb
        count = 0
        while sampled_bkg < start_sample_nb + nb_bkg:
            x = rng.randint(int(patch_size/2), img.shape[1] - int(patch_size/2))
            y = rng.randint(int(patch_size/2), img.shape[0] - int(patch_size/2))
            if not overlap_patch_roi((x,y), patch_size, roi_mask, cutoff=neg_cutoff):
                bkg = img[y - int(patch_size/2):y + int(patch_size/2), 
                            x - int(patch_size/2):x + int(patch_size/2)]


                sampled_bkg += 1

                if sampled_bkg ==1:

                    bkg_s_path = os.path.join(folder_s, "bkg_" + str(index) + ".png")
                    bkg= bkg.astype('uint16')
                    image_as_png(bkg,bkg_s_path)
                    
                    df_s.loc[index] = [bkg_s_path, 0]
                    

                else:

                    bkg_s10_path = os.path.join(folder_s, f"bkg_{index}_{sampled_bkg}.png")
    
                    bkg= bkg.astype('uint16')
                    image_as_png(bkg,bkg_s10_path)
                    df_s10.loc[index] = [bkg_s10_path, 0]


                if verbose:
                    print("sampled a bkg patch at (x,y) center=", (x,y))
                    sys.stdout.flush()
                    
    else:
        print("mask and img have different shape")
