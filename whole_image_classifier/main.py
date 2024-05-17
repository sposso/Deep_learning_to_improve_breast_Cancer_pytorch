import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np 
import torch.optim as optim
import torch.nn as nn
import datetime
from sklearn.model_selection import train_test_split
import time
import copy
from imageio import imread
import random

from tools import initialize_data_loader,Initialize_model,CBIS_MAMMOGRAM


def train_model(model, dataloaders, criterion, stages):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for stage in stages:
        
        if stage == "First_Stage":
            num_epochs = 20
            print("Params to update in the first stage ")
            params_to_update_first = []
            for name, param in model.named_parameters():
                if name.startswith('layer') or name.startswith("fc"):
                    param.requires_grad = True
                    params_to_update_first.append(param)
                    print("\t",name)
                else:
                    param.requires_grad = False
            optimizer = optim.Adam(params_to_update_first, lr = 1e-4, weight_decay=1e-4)
                    
            
            
        elif stage == "Second_Stage":
            num_epochs = 30
            print("Params to update in the second stage")
            params_to_update_second = []
            for name, param in model.named_parameters():
               
                param.requires_grad = True
                params_to_update_second.append(param)
                print("\t",name)
               
                    
            optimizer = optim.Adam(params_to_update_second, lr = 1e-5, weight_decay=0.001)
        
        for epoch in trange(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in tqdm(dataloaders[phase]):
                    inputs = inputs.to(device = device, dtype = torch.float)
                    labels = labels.to(device)

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

                print('Stage: {} Phase: {} Loss: {:.4f} Acc: {:.4f}'.format(stage, phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    return best_acc, best_model_wts

def test_function(model, best_model_wts,batch_size,num_classes,test_loader,loss_fn)

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    test_loss = 0.0
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    model.eval()
    predictions = []
    y_true =[]
    with torch.no_grad(): # prep model for evaluation

        for data, target in test_loader:
            if len(target.data) != batch_size:
                break
            # forward pass: compute predicted outputs by passing inputs to the model
            y_true.append(target)
            data = data.to(device=device, dtype =  torch.float)
            target = target.to(device= device)
            output = model(data)
            # calculate the loss
            loss = loss_fn(output, target)
            # update test loss 
            test_loss += loss.item()*data.size(0)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            predictions.append(pred.cpu())
            # compare predictions to true label
            correct = np.squeeze(pred.eq(target.data.view_as(pred)))
            # calculate test accuracy for each object class
            for i in range(batch_size):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

        # calculate and print avg test loss
        test_loss = test_loss/len(test_loader.dataset)
        print('Test Loss: {:.6f}\n'.format(test_loss))

        for i in range(2):
            if class_total[i] > 0:
                print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    str(i), 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
                print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

        print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))
         
        test_acc =  100. * np.sum(class_correct) / np.sum(class_total)
        
    y_pred = torch.vstack(predictions)
    y_p = torch.flatten(y_pred)
    y_t = torch.flatten(torch.vstack(y_true))
    AUC = roc_auc_score(y_t.numpy(),y_p.numpy())
    
    print('AUC: {:.6f}\n'.format(AUC))
    
    return test_acc, AUC
