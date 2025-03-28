import os
import torch
from torch import nn
from torchvision import models
import pytorch_lightning as pl

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

def initialize_patch_model(model_name, num_classes,use_pretrained = False, root = None):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None


    if model_name == "resnet":
        """ Resnet50
        """
        model_ft = models.resnet50()
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        
        if use_pretrained: 

            model_ft.load_state_dict(torch.load(os.path.join(root,'trained_models/best_s10_patch_model.pt'),
                                                map_location=torch.device('cpu')))
            
            print('Whole image classifier initialized with the s10 patch classifier weights')


    return model_ft


class Resnet50(nn.Module):
    def __init__(self,block,layers,use_pretrained,root, stride =1, inplanes = 2048, num_classes= 2):
        super(Resnet50, self).__init__()
        self.model = initialize_patch_model("resnet", 5, use_pretrained,root)
        print('Whole image classifier initialized with the s10 patch classifier weights')
        self.model = nn.Sequential(*list(self.model.children())[:-2])



        self.layer1 = _make_layer(inplanes, block, 512, layers[0],stride)
        self.layer2 = _make_layer(1024,block,512, layers[1],stride)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        #self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512*block.expansion,num_classes)






    def forward(self, x):
        x = self.model(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)


        return x

class PL_model(pl.LightningModule):
    def __init__(self, backbone):
        super().__init()
        
        self.backbone = backbone
        
    def forward(self, x):
        return self.backbone(x)