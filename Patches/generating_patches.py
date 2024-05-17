import numpy as np
import cv2
import pandas as pd
from imageio import imread
import png
from scipy import ndimage
import sys


from tools import image_as_png,crop_val,add_img_margins,segment_breast,overlap_patch_roi



def sample_patches(img_,index,label, roi_mask_,patch_size=224,
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
        
        pathi = ".../Breast_cancer/patches_256/s/" +"roi_"+ str(index) + ".png"
        image_as_png(s_patch,pathi)
        s_path.append(pathi)
        s_class.append(label)


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
            # import pdb; pdb.set_trace()

            if overlap_patch_roi((x,y), patch_size, roi_mask, cutoff=pos_cutoff):
                patch = img[y - int(patch_size/2):y + int(patch_size/2), 
                            x - int(patch_size/2):x + int(patch_size/2)]

                path_s10 = "...patches_256/s10/" +"roi_"+ str(index)+"_"+str(sampled_abn)+".png"
                patch= patch.astype('uint16')
                
                image_as_png(patch,path_s10)
                s10_path.append(path_s10)
                s10_class.append(label)
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

                    bkg_s_path = "...patches_256/s/" +"bkg_"+ str(index) +".png"
                    bkg= bkg.astype('uint16')
                    image_as_png(bkg,bkg_s_path)
                    s_path.append(bkg_s_path)
                    s_class.append(0)

                else:

                    bkg_s_path = "...patches_256/s10/" +"bkg_"+ str(index)+"_"+str(sampled_bkg)+".png"
                    bkg= bkg.astype('uint16')
                    image_as_png(bkg,bkg_s_path)
                    s10_path.append(bkg_s_path)
                    s10_class.append(0)


                if verbose:
                    print("sampled a bkg patch at (x,y) center=", (x,y))
                    sys.stdout.flush()
                    
    else:
        print("mask and img have different shape")
