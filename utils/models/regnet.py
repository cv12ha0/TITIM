'''
    Designing Network Design Spaces

    code:
        https://github.com/kuangliu/pytorch-cifar/blob/master/models/regnet.py
'''

import torch
from torch import nn


__all__ = [
    "RegNet", 
    "regnetx200mf", 
    "regnetx400mf", 
    "regnety400mf", 
]


def regnetx200mf(img_shape, num_classes=1000):
    model = RegNetX_200MF(img_shape, num_classes, )
    model.config.update({'model': 'RegNetX_200MF', })
    return model


def regnetx400mf(img_shape, num_classes=1000):
    model = RegNetX_400MF(img_shape, num_classes, )
    model.config.update({'model': 'RegNetX_400MF', })
    return model


def regnety400mf(img_shape, num_classes=1000):
    model = RegNetY_400MF(img_shape, num_classes, )
    model.config.update({'model': 'RegNetY_400MF', })
    return model



class RegNetSE(nn.Module):
    '''Squeeze-and-Excitation block.'''

    def __init__(self, in_planes, se_planes):
        super().__init__()
        self.se1 = nn.Conv2d(in_planes, se_planes, kernel_size=1, bias=True)
        self.se2 = nn.Conv2d(se_planes, in_planes, kernel_size=1, bias=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.avgpool(out)
        out = self.relu(self.se1(out))
        out = self.se2(out).sigmoid()
        out = x * out
        return out


class RegNetBlock(nn.Module):
    def __init__(self, w_in, w_out, stride, group_width, bottleneck_ratio, se_ratio):
        super().__init__()
        # 1x1
        w_b = int(round(w_out * bottleneck_ratio))
        self.conv1 = nn.Conv2d(w_in, w_b, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(w_b)
        # 3x3
        num_groups = w_b // group_width
        self.conv2 = nn.Conv2d(w_b, w_b, kernel_size=3,
                               stride=stride, padding=1, groups=num_groups, bias=False)
        self.bn2 = nn.BatchNorm2d(w_b)
        # se
        self.with_se = se_ratio > 0
        if self.with_se:
            w_se = int(round(w_in * se_ratio))
            self.se = RegNetSE(w_b, w_se)
        # 1x1
        self.conv3 = nn.Conv2d(w_b, w_out, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(w_out)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or w_in != w_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(w_in, w_out,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(w_out)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        if self.with_se:
            out = self.se(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class RegNet(nn.Module):
    def __init__(self, cfg, img_shape=(3, 224, 224), num_classes=1000):
        super(RegNet, self).__init__()
        self.cfg = cfg
        self.in_planes = 64
        self.c, self.h, self.w = img_shape
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.config = {'model': 'RegNet', 'img_shape': self.img_shape, 'num_classes': self.num_classes, }

        self.conv1 = nn.Conv2d(self.c, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(0)
        self.layer2 = self._make_layer(1)
        self.layer3 = self._make_layer(2)
        self.layer4 = self._make_layer(3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.cfg['widths'][-1], num_classes)

    def _make_layer(self, idx):
        depth = self.cfg['depths'][idx]
        width = self.cfg['widths'][idx]
        stride = self.cfg['strides'][idx]
        group_width = self.cfg['group_width']
        bottleneck_ratio = self.cfg['bottleneck_ratio']
        se_ratio = self.cfg['se_ratio']

        layers = []
        for i in range(depth):
            s = stride if i == 0 else 1
            layers.append(RegNetBlock(self.in_planes, width,
                                s, group_width, bottleneck_ratio, se_ratio))
            self.in_planes = width
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


def RegNetX_200MF(img_shape=(3, 224, 224), num_classes=1000):
    cfg = {
        'depths': [1, 1, 4, 7],
        'widths': [24, 56, 152, 368],
        'strides': [1, 1, 2, 2],
        'group_width': 8,
        'bottleneck_ratio': 1,
        'se_ratio': 0,
    }
    return RegNet(cfg, img_shape, num_classes)


def RegNetX_400MF(img_shape=(3, 224, 224), num_classes=1000):
    cfg = {
        'depths': [1, 2, 7, 12],
        'widths': [32, 64, 160, 384],
        'strides': [1, 1, 2, 2],
        'group_width': 16,
        'bottleneck_ratio': 1,
        'se_ratio': 0,
    }
    return RegNet(cfg, img_shape, num_classes)


def RegNetY_400MF(img_shape=(3, 224, 224), num_classes=1000):
    cfg = {
        'depths': [1, 2, 7, 12],
        'widths': [32, 64, 160, 384],
        'strides': [1, 1, 2, 2],
        'group_width': 16,
        'bottleneck_ratio': 1,
        'se_ratio': 0.25,
    }
    return RegNet(cfg, img_shape, num_classes)
