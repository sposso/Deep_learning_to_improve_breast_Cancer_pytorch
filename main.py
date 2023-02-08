import argparse
import os
import random
import shutil
import time
import warnings
from datetime import timedelta
import copy
import numpy as np
import pandas as pd 
from tqdm import tqdm

import torch
import torch.nn as nn 
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torchvision.transforms as T
import torchvision.datasets as datasets
import torchvision.models as models
from torch.nn.parallel import DistributedDataParallel
#from sklearn.metrics import roc_curve
#from sklearn.metrics import roc_auc_score
from PIL import Image

from utils.classifier import Bottleneck, Resnet50
from utils.train_util import initialize_data_loader, Initialize_model,first_stage,second_stage, save_checkpoint,load_checkpoint




def parse_args():
    parser = argparse.ArgumentParser(description='train, resume, test arguments')
    parser.add_argument('--project_root', default= os.getcwd())
    parser.add_argument('--batch_size', '-b',default=32, type = int, help = "mini-batch size per worker(GPU)" )
    parser.add_argument('--workers', '-w', default= 8, type = int, help ="Number of data loading workers")
    parser.add_argument("--checkpoint-file",default=os.getcwd()+"/tmp/checkpoint.pth.tar",type=str,help="checkpoint file path, to load and save to")
    parser.add_argument('--epochs','-e', default = 50, type = int, help='number of total epochs to run')


    return parser.parse_args()



def main():


    args = parse_args()
    device_id = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device_id)
    print(f"=> set cuda device = {device_id}")


   #initilaizes the default process group
    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(seconds=10)
    )


    # define loss function (criterion) and optimizer
    model, criterion = Initialize_model(device_id,args.project_root)


    first_optimizer = first_stage(model)
    second_optimizer = second_stage(model)

    train_loader,val_loader, test_loader = initialize_data_loader(args.batch_size,args.workers,args.project_root)

    dataloaders_dict = {'train':train_loader,'val': val_loader}


    state = load_checkpoint(args.checkpoint_file, device_id, model, second_optimizer)

    

    since = time.time()


    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        state.epoch = epoch
        train_loader.batch_sampler.sampler.set_epoch(epoch)

        if epoch <30:

            acc = train_model(model, dataloaders_dict, criterion,first_optimizer, device_id)

        else:

            acc = train_model(model, dataloaders_dict, criterion,second_optimizer, device_id)
        
        is_best = acc>state.best_acc1
        state.best_acc1 = max(acc,state.best_acc1)
        
        if device_id ==0:
            save_checkpoint(state, is_best, args.checkpoint_file)
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(state.best_acc1))



def train_model(model, dataloaders, criterion,optimizer,device_id):
    
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for img_tensor,labels in tqdm(dataloaders[phase]):


            inputs= img_tensor.expand(-1,3,*img_tensor.shape[2:])
            inputs = inputs.cuda(device_id, non_blocking=True)

            labels = labels.cuda(device_id, non_blocking=True)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                # Get model outputs and calculate loss
                # Special case for inception because in training it has an auxiliary output. In train
                #   mode we calculate the loss by summing the final output and the auxiliary output
                #   but in testing we only consider the final output


                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

        print('Phase: {} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        
        if phase == 'val':
            
            val_acc = epoch_acc


        print()
    
    return val_acc



if __name__ == '__main__':
    main()
