mport matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
import cv2 as cv
import pandas as pd
import os
import png

from resize_function import resize_image
from patches_utils import segment_breast

dataset = 'test'  # or 'testing'

df = pd.read_csv(f'final_datasets/{dataset}_dataset.csv')

# Create a folder to save resized images if it doesn't exist
base_folder = '/home/sposso22/work/shared_data/breast_cancer'
output_folder = os.path.join(base_folder, f'resized_images_{dataset}')

area_mask_pre_resize = np.zeros(df.shape[0])
area_mask_post_resize = np.zeros(df.shape[0])


# Create an empty dataframe to store the paths of the resized images
resized_images_list = []



if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for index, row in df.iterrows():

    print('*************************************************************')
    print(f'Processing image {index+1}/{df.shape[0]}')
    print('*************************************************************')

    
    img_path = row['img_path']
    mask_path = row['roi_path']
    
    # Read the DICOM images
    image = dicom.dcmread(img_path).pixel_array
    mask = dicom.dcmread(mask_path).pixel_array
   
    
    # Resize the image and mask
    resized_image = resize_image(img_path, 1152, 896)
    resized_mask = resize_image(mask_path, 1152, 896)
    
    # Segment the breast region from the resized image
    resized_image, resized_mask = segment_breast(resized_image, resized_mask, crop= False)
    
    # Ensure the resized image and mask are of the same shape
    if resized_image.shape != resized_mask.shape:
        raise ValueError(f'Resized image and mask shapes do not match: {resized_image.shape} vs {resized_mask.shape}')
    
    # Calculate the area of the mask before resizing
    area_mask_pre_resize[index] = np.sum(mask > 0)
    # Calculate the area of the mask after resizing
    area_mask_post_resize[index] = np.sum(resized_mask > 0)
    
    #save the resized images as png files
    resized_image_path = os.path.join(output_folder, f'resized_image_{index}.png')
    resized_mask_path = os.path.join(output_folder, f'resized_mask_{index}.png')

    with open(resized_image_path, 'wb') as f:
        writer = png.Writer(width=resized_image.shape[1], height=resized_image.shape[0],
                            bitdepth=16, greyscale=True)

        writer.write(f, resized_image.tolist())

    with open(resized_mask_path, 'wb') as f:
        writer = png.Writer(width=resized_mask.shape[1], height=resized_mask.shape[0],
                            bitdepth=16, greyscale=True)

        writer.write(f, resized_mask.tolist())
    print(f'Resized image saved to {resized_image_path}')
    print(f'Resized mask saved to {resized_mask_path}')
    # Append the paths and label to the resized images dataframe
   
    resized_images_df = resized_images_list.append({
        'img_path': resized_image_path,
        'mask_path': resized_mask_path,
        'label': row['label']
    })
    
# Save area mask pre and post resize to a numpy file npz 

np.savez(f'final_datasets/area_mask_pre_post_resize_{dataset}.npz',
         area_mask_pre_resize=area_mask_pre_resize,
         area_mask_post_resize=area_mask_post_resize)

# Save the resized images dataframe to a CSV file

resized_images_df = pd.DataFrame(resized_images_list)
resized_images_df.to_csv(f'final_datasets/{dataset}_resized_images.csv', index=False)

# Check if there are any resized masks with zero area

pre_resize = area_mask_pre_resize == 0
post_resize = area_mask_post_resize == 0    

print(f'Number of masks with zero area before resizing: {np.sum(pre_resize)}')
print(f'Number of masks with zero area after resizing: {np.sum(post_resize)}')
