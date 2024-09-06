'''
    Wide Residual Networks (https://arxiv.org/abs/1605.07146)

    code:
        https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/wideresidual.py
'''
import torch
from torch import nn
__all__ = ["WideResNet", "wide_resnet", ]


def wide_resnet(img_shape, num_classes=1000, depth=40, widen_factor=10):
    # Table 9: Best WRN performance over various datasets, single run results.
    model = WideResNet(WideBasic, depth, widen_factor, img_shape, num_classes)
    model.config.update({'model': 'WideResNet', })
    return model


class WideBasic(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.shortcut = nn.Sequential()

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride)
            )

    def forward(self, x):
        residual = self.residual(x)
        shortcut = self.shortcut(x)
        return residual + shortcut


class WideResNet(nn.Module):
    def __init__(self, block, depth=50, widen_factor=1, img_shape=(3, 224, 224), num_classes=1000):
        super().__init__()
        k = widen_factor
        l = int((depth - 4) / 6)
        self.in_channels = 16
        self.depth = depth
        self.c, self.h, self.w = img_shape
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.config = {'model': 'WideResNet', 'depth': depth, 'widen_factor': widen_factor, 
                       'img_shape': self.img_shape, 'num_classes': self.num_classes, }

        self.init_conv = nn.Conv2d(self.c, self.in_channels, 3, 1, padding=1)
        self.conv2 = self._make_layer(block, 16 * k, l, 1)
        self.conv3 = self._make_layer(block, 32 * k, l, 2)
        self.conv4 = self._make_layer(block, 64 * k, l, 2)
        self.bn = nn.BatchNorm2d(64 * k)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64 * k, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)
