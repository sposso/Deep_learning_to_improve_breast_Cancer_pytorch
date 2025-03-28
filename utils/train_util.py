from typing import Tuple
import os
import torch
from torch.utils.data import  DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from utils.tools import CBIS_MAMMOGRAM,MyIntensityShift
from  torch import  nn
from torch import optim

from utils.classifier import Bottleneck, Resnet50


#Two training stages 
def first_stage(model):
    
    print("Params to update in the first stage ")
    params_to_update_first = []
    for name, param in model.named_parameters():
        if name.startswith('layer') or name.startswith("fc"):
            param.requires_grad = True
            params_to_update_first.append(param)
            print("\t",name)
        else:
            param.requires_grad = False
    first_optimizer = optim.Adam(params_to_update_first, lr = 1e-4, weight_decay=1e-4)
    
    return first_optimizer

def second_stage(model):


    print("Params to update in the second stage")
    params_to_update_second = []
    for name, param in model.named_parameters():

        param.requires_grad = True
        params_to_update_second.append(param)
        print("\t",name)

    second_optimizer = optim.Adam(params_to_update_second, lr = 1e-5, weight_decay=0.001)
    
    return second_optimizer



def initialize_data_loader(batch_size,workers,root,aug = False) -> Tuple[DataLoader, DataLoader, DataLoader]:

    
    train = os.path.join(root,"data/train.csv")
    validation =os.path.join(root,"data/validation.csv")
    test = os.path.join(root,"data/test.csv")
   

    normalize = T.Normalize(mean=[0.2006],
                                     std=[0.2622])

    augmentation = T.Compose([T.ToTensor(),normalize,T.RandomHorizontalFlip(), T.RandomVerticalFlip(),T.RandomRotation(degrees=25),
                        T.RandomAffine(degrees=0, scale=(0.8, 0.99)),T.RandomResizedCrop(size=(1152,896),scale=(0.8,0.99)),
                        MyIntensityShift(shift= [80,120]), T.RandomAffine(degrees=0, shear=12)])


    if aug:
        train_dataset= CBIS_MAMMOGRAM(train, transform = augmentation)
    
    else: 
        train_dataset = CBIS_MAMMOGRAM(train, transform = T.Compose([T.ToTensor(),normalize]))
    #Normalizing the validation set
    validation_dataset = CBIS_MAMMOGRAM(validation, transform = T.Compose([T.ToTensor(),normalize]))
    #Normalizing the test set
    test_dataset = CBIS_MAMMOGRAM(test, transform = T.Compose([T.ToTensor(),normalize]))

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

def initialize_whole_model(root):
    
    model = Resnet50(Bottleneck, layers=[2,2], use_pretrained=True, root=root)

    criterion = nn.CrossEntropyLoss()
    
    return model, criterion