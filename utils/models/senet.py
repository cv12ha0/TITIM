"""
    The winner of ImageNet-2017.
    Squeeze-and-Excitation Networks (https://arxiv.org/abs/1709.01507)

    code:
        https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/senet.py
        https://github.com/kuangliu/pytorch-cifar/blob/master/models/senet.py
"""
import torch
from torch import nn

__all__ = [
    "SENet", 
    "senet18", 
    "senet34", 
    "senet50", 
    "senet101", 
    "senet152", 
]


def senet18(img_shape, num_classes=1000):
    model = SENet(SENetBlock, [2, 2, 2, 2], img_shape, num_classes)
    model.config.update({'model': 'SENet18', })
    return model


def senet34(img_shape, num_classes=1000):
    model = SENet(SENetBlock, [3, 4, 6, 3], img_shape, num_classes)
    model.config.update({'model': 'SENet34', })
    return model


def senet50(img_shape, num_classes=1000):
    model = SENet(SENetBottleneck, [3, 4, 6, 3], img_shape, num_classes)
    model.config.update({'model': 'SENet50', })
    return model


def senet101(img_shape, num_classes=1000):
    model = SENet(SENetBottleneck, [3, 4, 23, 3], img_shape, num_classes)
    model.config.update({'model': 'SENet101', })
    return model


def senet152(img_shape, num_classes=1000):
    model = SENet(SENetBottleneck, [3, 4, 36, 3], img_shape, num_classes)
    model.config.update({'model': 'SENet152', })
    return model




class SENetBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, r=16, squeeze='conv'):
        super().__init__()
        self.squeeze = squeeze
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes*self.expansion, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes*self.expansion)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*self.expansion)
            )

        # SE layers
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if squeeze in ['conv']:
            # Use nn.Conv2d instead of nn.Linear
            self.fc1 = nn.Conv2d(planes*self.expansion, planes*self.expansion//r, kernel_size=1)
            self.fc2 = nn.Conv2d(planes*self.expansion//r, planes*self.expansion, kernel_size=1)
        elif squeeze in ['linear']:
            self.fc1 = nn.Linear(planes*self.expansion, planes*self.expansion//r)
            self.fc2 = nn.Linear(planes*self.expansion//r, planes*self.expansion)
        else: 
            raise Exception("SENet: Unknown squeeze type ", squeeze)

    def forward(self, x):
        shortcut = self.shortcut(x)
        # Residual
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))  # self.bn2(self.conv2(out))
        
        # Squeeze
        w = self.avgpool(out)
        if self.squeeze in ['linear']:
            w = w.view(w.size(0), -1)
        # Excitation
        w = self.relu(self.fc1(w))
        w = self.sigmoid(self.fc2(w))
        w = w.view(out.size(0), out.size(1), 1, 1)  # .expand_as(out)

        out = out * w + shortcut
        out = self.relu(out)
        return out


class SENetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, r=16, squeeze='linear'):
        super().__init__()
        self.squeeze = squeeze
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes*self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes*self.expansion, kernel_size=1, stride=stride, bias=False), 
                nn.BatchNorm2d(planes*self.expansion)
            )

        # SE layers
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if squeeze in ['conv']:
            # Use nn.Conv2d instead of nn.Linear
            self.fc1 = nn.Conv2d(planes*self.expansion, planes*self.expansion//r, kernel_size=1)
            self.fc2 = nn.Conv2d(planes*self.expansion//r, planes*self.expansion, kernel_size=1)
        elif squeeze in ['linear']:
            self.fc1 = nn.Linear(planes*self.expansion, planes*self.expansion//r)
            self.fc2 = nn.Linear(planes*self.expansion//r, planes*self.expansion)
        else: 
            raise Exception("SENet: Unknown squeeze type ", squeeze)

    def forward(self, x):
        # out = self.relu(self.bn1(x))
        # shortcut = self.shortcut(out)
        # out = self.conv1(out)
        # out = self.conv2(self.relu(self.bn2(out)))
        shortcut = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))

        # Squeeze
        w = self.avgpool(out)
        if self.squeeze in ['linear']:
            w = w.view(w.size(0), -1)
        # Excitation
        w = self.relu(self.fc1(w))
        w = self.sigmoid(self.fc2(w))
        w = w.view(out.size(0), out.size(1), 1, 1)  # .expand_as(out)

        out = out * w + shortcut
        out = self.relu(out)
        return out


class SENet(nn.Module):
    def __init__(self, block, num_blocks, img_shape=(3, 224, 224), num_classes=1000):
        super().__init__()
        self.in_planes = 64
        self.c, self.h, self.w = img_shape
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.config = {'model': 'SENet', 'img_shape': self.img_shape, 'num_classes': self.num_classes, }

        self.conv1 = nn.Conv2d(self.c, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


