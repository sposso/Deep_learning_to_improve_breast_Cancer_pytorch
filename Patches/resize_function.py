import pydicom as dicom 
import cv2 as cv
import numpy as np


def resize_image(image_path,Height_new, Width_new):
    """
    Resize the image to the specified height and width while preserving the aspect ratio.
    
    Args:
        image_path (str): Path to the DICOM image file.
        Height_new (int): Desired height of the resized image.
        Width_new (int): Desired width of the resized image.
        
    Returns:
        np.ndarray: Resized image.
    """
    # Read the DICOM image
    image = dicom.dcmread(image_path).pixel_array
    
    print(f'Original image shape: {image.shape}')
    original_ratio = image.shape[1] / image.shape[0]
    
    
    # Resized based on the height

    scaling_factor = Height_new / image.shape[0]
    new_size = (int(image.shape[1] * scaling_factor), Height_new)
    
    # Double-check the aspect ratio
    new_aspect_ratio = new_size[0] / new_size[1]
    print(f'New aspect ratio: {new_aspect_ratio:.2f}')
    print(f'Original aspect ratio: {original_ratio:.2f}')
    resized_image = cv.resize(image, new_size, interpolation=cv.INTER_LINEAR)
    
    if resized_image.shape[1] > Width_new:
        # crop the image to the desired width
        crop_width = resized_image.shape[1] - Width_new
        
        print(f'Crop width: {crop_width}')
        
        # Half of the crop on each side
        left_crop = crop_width // 2
        right_crop = crop_width - left_crop
        
        print(f'Left crop: {left_crop}, Right crop: {right_crop}')
        
        # Apply the crop
        resized_image = resized_image[:, left_crop:resized_image.shape[1]-right_crop]
        
        print(f'Cropped image shape: {resized_image.shape}')
        
    elif resized_image.shape[1] < Width_new:
        # pad the image to the desired width
        pad_width = Width_new - resized_image.shape[1]
        print(f'Pad width: {pad_width}')
        # Half of the pad on each side
        left_pad = pad_width // 2
        right_pad = pad_width - left_pad
        print(f'Left pad: {left_pad}, Right pad: {right_pad}')
        # Apply the padding
        resized_image = cv.copyMakeBorder(resized_image, 0, 0, left_pad, right_pad, cv.BORDER_CONSTANT, value=0)
        print(f'Padded image shape: {resized_image.shape}')
        
    else:
        # Image is already in the desired aspect ratio
        print('Image is already in the desired aspect ratio.')
        resized_image = cv.resize(image, (Width_new, Height_new), interpolation=cv.INTER_LINEAR)
        print(f'Resized image shape: {resized_image.shape}')
        
    # raise an error if the resized image is not in in the desired shape 
    if resized_image.shape[0] != Height_new or resized_image.shape[1] != Width_new:
        raise ValueError(f'Resized image shape {resized_image.shape} does not match the desired shape ({Height_new}, {Width_new})')
        
        
    return resized_image
