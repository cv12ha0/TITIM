'''
    ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices (https://arxiv.org/abs/1707.01083v2)

    code:
        https://github.com/kuangliu/pytorch-cifar/blob/master/models/shufflenet.py
        https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/shufflenet.py
'''

import torch
from torch import nn

__all__ = [
    "ShuffleNet", 
    "shufflenetg1", 
    "shufflenetg2", 
    "shufflenetg3", 
]


def shufflenetg1(img_shape, num_classes=1000):
    model = ShuffleNet([144, 288, 567], [4, 8, 4], 1, img_shape, num_classes, )
    model.config.update({'model': 'ShuffleNetG1', })
    return model

def shufflenetg2(img_shape, num_classes=1000):
    model = ShuffleNet([200, 400, 800], [4, 8, 4], 2, img_shape, num_classes, )
    model.config.update({'model': 'ShuffleNetG2', })
    return model

def shufflenetg3(img_shape, num_classes=1000):
    model = ShuffleNet([240, 480, 960], [4, 8, 4], 3, img_shape, num_classes, )
    model.config.update({'model': 'ShuffleNetG3', })
    return model

def shufflenetg4(img_shape, num_classes=1000):
    model = ShuffleNet([272, 544, 1088], [4, 8, 4], 4, img_shape, num_classes, )
    model.config.update({'model': 'ShuffleNetG4', })
    return model

def shufflenetg8(img_shape, num_classes=1000):
    model = ShuffleNet([384, 768, 1536], [4, 8, 4], 8, img_shape, num_classes, )
    model.config.update({'model': 'ShuffleNetG8', })
    return model


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C//g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


class ShuffleNetBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups):
        super().__init__()
        self.stride = stride

        mid_planes = out_planes//4
        g = 1 if in_planes == 24 else groups
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.shuffle1 = ShuffleBlock(groups=g)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = self.shortcut(x)
        out = self.relu(torch.cat([out, res], 1)) if self.stride == 2 else self.relu(out+res)
        return out


class ShuffleNet(nn.Module):
    def __init__(self, out_planes, num_blocks, groups, img_shape=(3, 224, 224), num_classes=1000):
        super().__init__()
        self.out_planes = out_planes
        self.num_blocks = num_blocks
        self.groups = groups
        self.c, self.h, self.w = img_shape
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.config = {'model': 'ShuffleNet', 'img_shape': self.img_shape, 'num_classes': self.num_classes, 'groups': self.groups}

        self.in_planes = 24
        self.conv1 = nn.Conv2d(self.c, 24, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
        self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
        self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(out_planes[2], num_classes)

    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            cat_planes = self.in_planes if i == 0 else 0
            layers.append(ShuffleNetBottleneck(self.in_planes, out_planes-cat_planes, stride=stride, groups=groups))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

