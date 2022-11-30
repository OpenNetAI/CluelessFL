import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import platform
import copy
import time
import torch.nn.functional as F

from torch.nn.modules.module import Module

from typing import Tuple, Union
from torch import Tensor

from collections import OrderedDict


####################### CNNNet class ######################
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # 定义前向传播过程，输入为x
    def forward(self, x, start_layer_idx = 0):
        if start_layer_idx < 0: #
            return self.mapping(x, start_layer_idx=start_layer_idx)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return x, out

    def mapping(self, z_input, start_layer_idx=-1):
        z = z_input # (32,32)
        # n_layers = 8
        # for layer_idx in range(n_layers + start_layer_idx, n_layers):
            # layer = self.layers[layer_idx]
        # z = self.fc2(z)
        z = self.fc3(z)
        # if self.output_dim > 1:
        out=F.log_softmax(z, dim=1)
        # result = {'output': out}
        return z, out


# 定义网络模型
class CIFARLeNet(nn.Module):
    def __init__(self):
        super(CIFARLeNet, self).__init__()

        # 卷积层
        self.cnn = nn.Sequential(
            # 卷积层1，3通道输入，6个卷积核，核大小5*5
            # 经过该层图像大小变为32-5+1，28*28
            # 经2*2最大池化，图像变为14*14
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 卷积层2，6输入通道，16个卷积核，核大小5*5
            # 经过该层图像变为14-5+1，10*10
            # 经2*2最大池化，图像变为5*5
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 全连接层
        self.fc = nn.Sequential(
            # 16个feature，每个feature 5*5
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, start_layer_idx=0):
        if start_layer_idx < 0: #
            return self.mapping(x, start_layer_idx=start_layer_idx)
        x = self.cnn(x)

        # x.size()[0]: batch size
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = self.fc(x)

        return x
    
    
    def mapping(self, z_input, start_layer_idx=-1):
        z = z_input # (32,32)
        # n_layers = 8
        # for layer_idx in range(n_layers + start_layer_idx, n_layers):
            # layer = self.layers[layer_idx]
        z = self.fc2(z)
        z = self.fc3(z)
        # if self.output_dim > 1:
        out=F.log_softmax(z, dim=1)
        # result = {'output': out}
        return out

####################### ResNet related  classes ######################
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.elu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.elu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.elu(self.bn1(self.conv1(x)))
        out = F.elu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.elu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, qualifier, num_classes=10):
        super(ResNet, self).__init__()
        self.qualifier=qualifier # 9 or 18
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear1 = nn.Linear(100*block.expansion, num_classes)
        self.linear2 = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, start_layer_idx = 0):
        if start_layer_idx < 0: #
            return self.mapping(x, start_layer_idx=start_layer_idx)
        out = F.elu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        final_out = self.linear2(out)
        return out, final_out
    
    def mapping(self, z_input, start_layer_idx=-1):
        z = z_input # (32,32)
        z = self.linear1(z)
        out = F.log_softmax(z, dim=1)
        return z,out

    # possible partition of layers, by index
    # if ci<= upperindex[k] and ci>upperindex[k-1], 
    # all parameters belong to partition k
    # NOTE: this should be specified by hand
    def train_order_block_ids(self):
      if self.qualifier==18:
       return [[0,2],[3,8],[9,14],[15,23],[24,29],[30,38],[39,44],[45,53],[54,59],[60,61]]
      else:
       return [[0,2],[3,8],[9,14],[15,17],[18,23],[24,29],[30,32],[33,37]]

    # return linear layer ids (empty)
    def linear_layer_ids(self):
      return []
 

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2], qualifier=18)

def ResNet9():
    return ResNet(BasicBlock, [1,1,1,1], qualifier=9)


####################### AlexNet class ######################
class AlexNet_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet_CIFAR10, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            #nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.out_layer = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        out = self.out_layer(x)
        return x, out
