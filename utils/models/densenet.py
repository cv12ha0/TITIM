import numpy as np
import torch
import torchvision
from torch import nn


def densenet_cifar(img_shape, num_classes=1000, growth_rate=12, reduction=0.5):
    model = DenseNet(Bottleneck, [6, 12, 24, 16], img_shape, num_classes, growth_rate, reduction)
    model.config.update({'model': 'DenseNet_cifar', })
    return model


def densenet121(img_shape, num_classes=1000, growth_rate=32, reduction=0.5):
    model = DenseNet(Bottleneck, [6, 12, 24, 16], img_shape, num_classes, growth_rate, reduction)
    model.config.update({'model': 'DenseNet121', })
    return model


def densenet169(img_shape, num_classes=1000, growth_rate=32, reduction=0.5):
    model = DenseNet(Bottleneck, [6, 12, 32, 32], img_shape, num_classes, growth_rate, reduction)
    model.config.update({'model': 'DenseNet169', })
    return model


def densenet201(img_shape, num_classes=1000, growth_rate=32, reduction=0.5):
    model = DenseNet(Bottleneck, [6, 12, 48, 32], img_shape, num_classes, growth_rate, reduction)
    model.config.update({'model': 'DenseNet201', })
    return model


def densenet264(img_shape, num_classes=1000, growth_rate=32, reduction=0.5):
    model = DenseNet(Bottleneck, [6, 12, 64, 18], img_shape, num_classes, growth_rate, reduction)
    model.config.update({'model': 'DenseNet264', })
    return model


def densenet161(img_shape, num_classes=1000, growth_rate=48, reduction=0.5):
    model = DenseNet(Bottleneck, [6, 12, 36, 24], img_shape, num_classes, growth_rate, reduction)
    model.config.update({'model': 'DenseNet161', })
    return model


class Bottleneck(nn.Module):
    """
    Bottleneck module in DenseNet Arch.
    See: https://arxiv.org/abs/1608.06993
    """
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        y = self.conv1(self.relu(self.bn1(x)))
        y = self.conv2(self.relu(self.bn2(y)))
        x = torch.cat([y, x], 1)
        return x


class Transition(nn.Module):
    """
    Transition module in DenseNet Arch.
    See: https://arxiv.org/abs/1608.06993
    """
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        x = torch.nn.functional.avg_pool2d(x, 2)
        return x


class DenseNet(nn.Module):
    """
    From: https://github.com/icpm/pytorch-cifar10/blob/master/models/DenseNet.py
    """
    def __init__(self, block, num_block, img_shape=(3, 224, 224), num_classes=1000, growth_rate=12, reduction=0.5, ):
        import math
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.reduction = reduction
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.config = {'model': 'DenseNet', 'img_shape': self.img_shape, 'num_classes': self.num_classes,
                       'growth_rate': growth_rate, 'reduction': reduction}

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, num_block[0])
        num_planes += num_block[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, num_block[1])
        num_planes += num_block[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, num_block[2])
        num_planes += num_block[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, num_block[3])
        num_planes += num_block[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.fc = nn.Linear(num_planes, num_classes)
        self.relu = nn.ReLU(True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def _make_dense_layers(self, block, in_planes, num_block):
        layers = []
        for i in range(num_block):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.trans3(self.dense3(x))
        x = self.dense4(x)
        x = self.avgpool(self.relu(self.bn(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
