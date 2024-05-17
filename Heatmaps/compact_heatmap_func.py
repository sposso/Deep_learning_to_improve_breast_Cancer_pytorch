''' This function sweep image data with the trained patch classifier to 
    heatmaps as well. Unlike the script "main.py", which contains the same function, this one is more efficient 
    since use only torch tensors'''
import cv2 
import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader


from tqdm import tqdm, trange

from tools import add_img_margins, sweep_img_patches,initialize_model
    
    def prob_heatmap(img_tensor, patch_size, stride, 
                     model,mean,std,device,batch_size):
    '''Sweep image data with a trained model to produce prob heatmaps
    '''
    
    nb_row = round(float(img.shape[0] - patch_size)/stride + 1)
    nb_col = round(float(img.shape[1] - patch_size)/stride + 1)
    nb_row = int(nb_row)
    nb_col = int(nb_col)
    print(nb_row)
    
    heatmap_list = []
    unfold = torch.nn.Unfold(kernel_size=(patch_size,patch_size), dilation=1, padding=112, stride=8)
    for img in tqdm(img_tensor):
        img= img.to(device=device)
        img = img.unsqueeze(dim=0)
        img_= unfold(img)
        img_ = img_.permute(0,2,1)
        n = img_.size(0)
        c = img_.size(1)
        img_ = img_.view(c,1,224,224)
        patch_X = img_.expand(-1,3,-1,-1)



        model.eval()
        with torch.no_grad():
            #patch_X = torch.tensor(patch_X,dtype = torch.float32)
            transform = T.Normalize(mean,std)
            patch_X = transform(patch_X.clone())
            loader = DataLoader(patch_X, batch_size)
            preds =[]


            for i in loader:
                i = i #.to(device)
                chunk_preds = model(i)
                chunk_preds = nn.functional.softmax(chunk_preds, dim=1)  
                preds.append(chunk_preds)

          
            pred = torch.vstack(preds)
            #print(pred.shape)
            pred = pred[:,1:5].sum(axis=1)
            print('holi')

            heatmap = pred.reshape((nb_row, nb_col))
            
            heatmap_list.append(heatmap.unsqueeze(dim=0))
        
        heatmap_tensor = torch.cat(heatmap_list)
            
    return heatmap_tensor.cpu().detach()
