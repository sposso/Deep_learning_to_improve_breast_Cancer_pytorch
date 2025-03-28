import os
import torch
from torch import nn
from torchvision import models
import pytorch_lightning as pl
import torch.distributed as dist
from utils.deformation import prob_heatmap_tensor, warped_imgs
import numpy as np
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import BaseFinetuning
class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def _make_layer(inplanes, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    model = nn.Sequential(*layers)

    for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    return model

def initialize_whole_model(root):
    model = Resnet50(Bottleneck, layers=[2,2], use_pretrained=True,root=root)
    return model

def initialize_patch_model(model_name, num_classes,set,use_pretrained = False, root = None):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.

    model_ft = None
    if model_name == "resnet":
        """ Resnet50"""
        model_ft = models.resnet50()
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        
        if use_pretrained and set=="s10":
            model_ft.load_state_dict(torch.load(os.path.join(root,'trained_models/best_s10_patch_model.pt'),
                                                map_location=torch.device('cpu')))

        elif usepretrained and set == "s"
            model_ft.load_state_dict(torch.load(os.path.join(root,'trained_models/best_s_patch_model.pt'),
                                                map_location=torch.device('cpu')))


    return model_ft

class Resnet50(nn.Module):
    def __init__(self,block,layers,set,use_pretrained,root, stride =1, inplanes = 2048, num_classes= 1):
        super(Resnet50, self).__init__()
        self.model = initialize_patch_model("resnet", 5,set, use_pretrained,root)
        self.model = nn.Sequential(*list(self.model.children())[:-2])

        self.layer1 = _make_layer(inplanes, block, 512, layers[0],stride)
        self.layer2 = _make_layer(1024,block,512, layers[1],stride)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        #self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512*block.expansion,num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class FullClassifier(pl.LightningModule):
    def __init__(self, args, res,set):
        super().__init__()

        self.patch_classifier = initialize_patch_model("resnet", num_classes=5,use_pretrained = True, root = args.project_root)
        self.patch_classifier.requires_grad = False

        self.backbone = initialize_whole_model(args.project_root)
        self.backbone.requires_grad = True

        self.res = res
        self.sigma = args.sigma
        self.criterion = nn.BCELoss()

        self.accuracy_metric = Accuracy(task='binary')
        self.max_accuracy = 0
 
    def forward(self, x):
        return self.backbone(x)
 
    def training_step(self, batch, batch_idx):
        x, y = batch 
        y = y.to(torch.float)

        
        #heatmaps = prob_heatmap_tensor(x, self.patch_classifier)
        #sampled_imgs = warped_imgs(x,heatmaps, self.res, self.sigma)

        inputs= x.expand(-1,3,*x.shape[2:])
        #inputs= sampled_imgs.expand(-1,3,*sampled_imgs.shape[2:])
        
        # compute loss
        preds = self.backbone(inputs).squeeze()
        loss = self.criterion(preds, y)
        
        self.log('train_loss', loss)
        return loss


    def validation_step(self, batch, batch_idx): 
        x, y = batch 
        y = y.to(torch.float)

        #heatmaps = prob_heatmap_tensor(x, self.patch_classifier)
        #sampled_imgs = warped_imgs(x,heatmaps, self.res, self.sigma)

        #inputs= sampled_imgs.expand(-1,3,*sampled_imgs.shape[2:])
        inputs= x.expand(-1,3,*x.shape[2:])

        # compute loss
        preds = self.forward(inputs).squeeze() 
        loss = self.criterion(preds, y)
        
        # compute accuracy
        acc =  self.accuracy_metric(preds, y)
        self.log('validation_loss', loss, on_epoch=True, sync_dist=True)
        self.log('accuracy', acc, on_epoch=True, prog_bar=True, sync_dist=True)

        return acc.item()
    
    def validation_epoch_end(self, outputs):
        if outputs:
            acc = np.mean(outputs)

            if acc > self.max_accuracy:
                self.max_accuracy = acc
                
            self.log('accuracy', acc, prog_bar=True, sync_dist=True)

        else:
            # needed for checkpointing
            self.log('accuracy', 0., prog_bar=True, sync_dist=True)
            
    def configure_optimizers(self):
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-4, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=100)
        
        return [optim], [scheduler]

class FeatureExtractorFreezeUnfreeze(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=30):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module):
        for name, param in pl_module.backbone.named_parameters():
            if name.startswith('layer') or name.startswith("fc"):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        # When `current_epoch` is 10, feature_extractor will start training.
        if current_epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=pl_module.backbone,
                optimizer=optimizer,
                train_bn=True,
        )