submodule2import torch
import torch.nn as nn
import torch.nn.functional as F

# Construct customized ResNet

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
 
        return out

class Tier2(nn.Module):
        
    def __init__(self, pcpt_block, pcpt_layers, scoop_block, scoop_layers, h, w, pcpt_is_upsample=0, scoop_is_upsample=0):
        self.inplanes = 64
        self.pcpt_is_upsample = pcpt_is_upsample
        super(Submodule2, self).__init__()
        self.pcpt_conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.pcpt_bn1 = nn.BatchNorm2d(64)
        self.pcpt_relu = nn.ReLU(inplace=True)
        self.pcpt_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pcpt_upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.pcpt_layer1 = self._make_layer(pcpt_block, 128, pcpt_layers[0])
        self.pcpt_layer2 = self._make_layer(pcpt_block, 256, pcpt_layers[1])
        self.pcpt_layer3 = self._make_layer(pcpt_block, 512, pcpt_layers[2])

        self.inplanes = 512
        self.scoop_is_upsample = scoop_is_upsample
        self.scoop_upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.scoop_layer1 = self._make_layer(scoop_block, 256, scoop_layers[0])
        self.scoop_layer2 = self._make_layer(scoop_block, 128, scoop_layers[1])
        self.scoop_layer3 = self._make_layer(scoop_block, 64, scoop_layers[2])
        self.scoop_conv1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.scoop_bn1 = nn.BatchNorm2d(1)
        self.scoop_conv2 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.scoop_bn2 = nn.BatchNorm2d(3)
        self.scoop_relu = nn.ReLU(inplace=True)
        self.x1_hidden = torch.nn.Linear(3*200*200, 200)
        self.x2_hidden = torch.nn.Linear(2, 200)
        self.x_hidden1 = torch.nn.Linear(400, 200)
        self.x_hidden2 = torch.nn.Linear(200, 50)
        self.x_hidden3 = torch.nn.Linear(50, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        x1 = self.pcpt_conv1(x1)
        x1 = self.pcpt_bn1(x1)
        x1 = self.pcpt_relu(x1)
        x1 = self.pcpt_maxpool(x1)

        x1 = self.pcpt_layer1(x1)
        x1 = self.pcpt_maxpool(x1)
        x1 = self.pcpt_layer2(x1)
        x1 = self.pcpt_layer3(x1)

        x1 = self.scoop_layer1(x1)
        x1 = self.scoop_layer2(x1)
        x1 = self.scoop_upsample(x1)
        x1 = self.scoop_layer3(x1)
        x1 = self.scoop_upsample(x1)

        x1 = self.scoop_conv2(x1)
        x1 = self.scoop_bn2(x1)
        x1 = self.scoop_relu(x1)       
        x1 = x1.reshape(x1.shape[0],-1)
        
        x1 = F.relu(self.x1_hidden(x1))
        
        x2 = F.relu(self.x2_hidden(x2))
        
        x = torch.cat((x1, x2), dim=-1)
        
        x = F.relu(self.x_hidden1(x))
        x = F.relu(self.x_hidden2(x))
        x = F.relu(self.x_hidden3(x))
        #x=torch.reshape(x,(-1,3,200))
        
        return x
