'''
    Progressive Neural Architecture Search

    code:
        https://github.com/kuangliu/pytorch-cifar/blob/master/models/pnasnet.py
'''

import torch
from torch import nn

__all__ = [
    "PNASNet", 
    "pnasneta", 
    "pnasnetb", 
]


def pnasneta(img_shape, num_classes=1000):
    model = PNASNetA(img_shape, num_classes, )
    model.config.update({'model': 'PNASNetA', })
    return model


def pnasnetb(img_shape, num_classes=1000):
    model = PNASNetB(img_shape, num_classes, )
    model.config.update({'model': 'PNASNetB', })
    return model


class PNASNetSepConv(nn.Module):
    '''Separable Convolution.'''
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding=(kernel_size-1)//2, bias=False, groups=in_planes)
        self.bn1 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        return self.bn1(self.conv1(x))


class PNASNetCellA(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.stride = stride
        self.sep_conv1 = PNASNetSepConv(in_planes, out_planes, kernel_size=7, stride=stride)
        if stride == 2:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(out_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        y1 = self.sep_conv1(x)
        y2 = self.maxpool(x)
        if self.stride == 2:
            y2 = self.bn1(self.conv1(y2))
        y = self.relu(y1 + y2)
        return y

class PNASNetCellB(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.stride = stride
        # Left branch
        self.sep_conv1 = PNASNetSepConv(in_planes, out_planes, kernel_size=7, stride=stride)
        self.sep_conv2 = PNASNetSepConv(in_planes, out_planes, kernel_size=3, stride=stride)
        # Right branch
        self.sep_conv3 = PNASNetSepConv(in_planes, out_planes, kernel_size=5, stride=stride)
        if stride == 2:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(out_planes)
        # Reduce channels
        self.conv2 = nn.Conv2d(2*out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Left branch
        y1 = self.sep_conv1(x)
        y2 = self.sep_conv2(x)
        # Right branch
        y3 = self.maxpool(x)
        if self.stride == 2:
            y3 = self.bn1(self.conv1(y3))
        y4 = self.sep_conv3(x)
        # Concat & reduce channels
        b1 = self.relu(y1+y2)
        b2 = self.relu(y3+y4)
        y = torch.cat([b1, b2], 1)
        y = self.relu(self.bn2(self.conv2(y)))
        return y

class PNASNet(nn.Module):
    def __init__(self, cell_type, num_cells, num_planes, img_shape=(3, 224, 224), num_classes=1000):
        super().__init__()
        self.in_planes = num_planes
        self.cell_type = cell_type
        self.c, self.h, self.w = img_shape
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.config = {'model': 'PNASNet', 'img_shape': self.img_shape, 'num_classes': self.num_classes, }

        self.conv1 = nn.Conv2d(self.c, num_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_planes)

        self.layer1 = self._make_layer(num_planes, num_cells=6)
        self.layer2 = self._downsample(num_planes*2)
        self.layer3 = self._make_layer(num_planes*2, num_cells=6)
        self.layer4 = self._downsample(num_planes*4)
        self.layer5 = self._make_layer(num_planes*4, num_cells=6)

        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_planes*4, num_classes)

    def _make_layer(self, planes, num_cells):
        layers = []
        for _ in range(num_cells):
            layers.append(self.cell_type(self.in_planes, planes, stride=1))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _downsample(self, planes):
        layer = self.cell_type(self.in_planes, planes, stride=2)
        self.in_planes = planes
        return layer

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def PNASNetA(img_shape=(3, 224, 224), num_classes=1000):
    return PNASNet(PNASNetCellA, num_cells=6, num_planes=44, img_shape=img_shape, num_classes=num_classes)

def PNASNetB(img_shape=(3, 224, 224), num_classes=1000):
    return PNASNet(PNASNetCellB, num_cells=6, num_planes=32, img_shape=img_shape, num_classes=num_classes)
