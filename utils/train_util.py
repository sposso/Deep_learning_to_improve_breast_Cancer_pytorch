from typing import Tuple
import os
import torch
from torch.utils.data import  DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from utils.tools import CBIS_MAMMOGRAM,MyIntensityShift
from  torch import  nn
from torch import optim


def initialize_data_loader(res,w,scale,batch_size,workers,root,aug,lambd) -> Tuple[DataLoader, DataLoader, DataLoader]:

    
    train = os.path.join(root,"data/train.csv")
    validation =os.path.join(root,"data/validation.csv")
    test = os.path.join(root,"data/test.csv")
   

    normalize = T.Normalize(mean=[0.2006],
                                     std=[0.2622])
    
    if aug is None:
        augmentation = T.Compose([normalize,T.resize(size=res)])

    elif aug == 'BIG':
        augmentation = T.Compose([normalize,T.RandomHorizontalFlip(), T.RandomVerticalFlip(),T.RandomRotation(degrees=25),
                            T.RandomAffine(degrees=0, scale=(0.8, 0.99)),T.resize(size=res),T.RandomResizedCrop(size=res,scale=(0.8,0.99)),
                            MyIntensityShift(shift= [80,120]), T.RandomAffine(degrees=0, shear=12)])
        

    
    elif aug == "SMALL": 
        augmentation = T.Compose([normalize,T.resize(size=res),T.RandomHorizontalFlip(), T.RandomVerticalFlip(),T.RandomRotation(degrees=25)])



    else:
        raise ValueError("Bad aug type")

    train_dataset = CBIS_MAMMOGRAM(train,res,w,scale,lambd, transform = augmentation)
    #Normalizing the validation set
    validation_dataset = CBIS_MAMMOGRAM(validation,res,w,scale,lambd, transform = normalize)
    #Normalizing the test set
    test_dataset = CBIS_MAMMOGRAM(test,res,w,scale,lambd, transform = normalize)

     # Restricts data loading to a subset of the dataset exclusive to the current process
    weights = torch.load(os.path.join(root,"data/sampler_weight.pt"))
    train_sampler = WeightedRandomSampler(weights, len(weights))
   
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)


    return train_loader,val_loader,test_loader