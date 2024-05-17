import cv2 
import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader


from tqdm import tqdm, trange

from tools import add_img_margins, sweep_img_patches,initialize_model

def my_prob_heatmap(img_tensor, patch_size, stride, 
                     model,mean,std,device,batch_size, data_format = 'channels_first'):
    
    '''Sweep image data with the trained patch classifier  to produce the heatmaps

    -mean and std corresponds to the mean and the standard deviation of the patch dataset employed to train the patch classifier
    -device : If you have a GPU availabe, you should use it to speed up the patch classification  process
    -batch size : NUmber of  patches that will be classified in one iteration"

    '''
   
    heatmap_list = []
    
    for img in tqdm(img_tensor):
        img = img.squeeze(dim=0)
        img = add_img_margins(img, int(patch_size/2))
        patch_dat, nb_row, nb_col = sweep_img_patches(
            img, patch_size, stride)
        


        if data_format == 'channels_first':
            patch_X = np.zeros((patch_dat.shape[0], 3, 
                                patch_dat.shape[1], 
                                patch_dat.shape[2]), 
                                dtype='float32')
            patch_X[:,0,:,:] = patch_dat
            patch_X[:,1,:,:] = patch_dat
            patch_X[:,2,:,:] = patch_dat
        else:
            patch_X = np.zeros((patch_dat.shape[0], 
                                patch_dat.shape[1], 
                                patch_dat.shape[2], 3), 
                                dtype='float32')
            patch_X[:,:,:,0] = patch_dat
            patch_X[:,:,:,1] = patch_dat
            patch_X[:,:,:,2] = patch_dat



        model.eval()
        with torch.no_grad():
            patch_X = torch.tensor(patch_X,dtype = torch.float32)
            transform = T.Normalize(mean,std)
            patch_X = transform(patch_X)
            loader = DataLoader(patch_X, batch_size)
            preds =[]


            for i in loader:
                i = i.to(device)
                chunk_preds = model(i)
                chunk_preds = nn.functional.softmax(chunk_preds, dim=1)  
                #print(chunk_preds)
                preds.append(chunk_preds)

          
            pred = torch.vstack(preds)

            #Get rid of the first channels, which corresponds to the background probability grid. 
            pred = pred[:,1:5].sum(axis=1)
            print(pred)
         
            heatmap = pred.reshape((nb_row, nb_col))
            
            heatmap_list.append(heatmap.unsqueeze(dim=0))
        
        heatmap_tensor = torch.cat(heatmap_list)
            
    return heatmap_tensor.cpu().detach()
