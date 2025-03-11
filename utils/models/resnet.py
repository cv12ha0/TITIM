import numpy as np
import torch
import torchvision
from torch import nn


def resnet18(img_shape, num_classes=1000, include_top=True):
    # pretrained weights:  https://download.pytorch.org/models/resnet18-5c106cde.pth
    model = ResNet(ResNetBuildingBlock, [2, 2, 2, 2], img_shape, num_classes, include_top)
    model.config.update({'model': 'ResNet18', })
    return model


def resnet34(img_shape, num_classes=1000, include_top=True):
    # pretrained weights:  https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model = ResNet(ResNetBuildingBlock, [3, 4, 6, 3], img_shape, num_classes, include_top)
    model.config.update({'model': 'ResNet34', })
    return model


def resnet50(img_shape, num_classes=1000, include_top=True):
    # pretrained weights:  https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model = ResNet(ResNetBottleNeckBlock, [3, 4, 6, 3], img_shape, num_classes, include_top)
    model.config.update({'model': 'ResNet50', })
    return model


def resnet101(img_shape, num_classes=1000, include_top=True):
    # pretrained weights:  https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model = ResNet(ResNetBottleNeckBlock, [3, 4, 23, 3], img_shape, num_classes, include_top)
    model.config.update({'model': 'ResNet101', })
    return model


def resnet152(img_shape, num_classes=1000, include_top=True):
    # pretrained weights:  https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model = ResNet(ResNetBottleNeckBlock, [3, 8, 36, 3], img_shape, num_classes, include_top)
    model.config.update({'model': 'ResNet152', })
    return model


class ResNetBuildingBlock(nn.Module):
    """
    Basic block in ResNet18, 34
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResNetBuildingBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:  # downsample residual branch
            identity = self.downsample(identity)
        conv1_out = self.relu(self.bn1(self.conv1(x)))  # conv - bn - relu
        conv2_out = self.bn2(self.conv2(conv1_out))
        out_add = conv2_out + identity  # conv + residual
        out = self.relu(out_add)
        return out


class ResNetBottleNeckBlock(nn.Module):
    """
    Basic block in ResNet50, 101, 152, ...
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResNetBottleNeckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)
        conv1_out = self.relu(self.bn1(self.conv1(x)))  # conv - bn - relu
        conv2_out = self.relu(self.bn2(self.conv2(conv1_out)))
        conv3_out = self.bn3(self.conv3(conv2_out))
        out_add = conv3_out + identity
        out = self.relu(out_add)

        return out


class ResNet(nn.Module):
    def __init__(self, block, block_num, img_shape=(3, 224, 224), num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.c, self.h, self.w = img_shape
        self.in_channels = 64  # initial conv, channel->64
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.config = {'model': 'ResNet', 'img_shape': self.img_shape, 'num_classes': self.num_classes,
                       'include_top': include_top}

        # down sample conv
        self.conv1 = nn.Conv2d(self.c, out_channels=self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        # (x-k+2p+1)/s+1，padding=3, img_size/=2
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        # padding=1,halve output size，when dilation=1，maxpool: (h+2*p-k)/s+1
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)

        # stride=2 (down sample since first conv)
        self.layer1 = self._make_layer(block, 64, block_num[0])
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, block_num[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.feature_pixels = 16
        self.feature_map_channels = 1024
        self.feature_vector_size = 2048
        self._feature_extractor = None  # self.features[0:-1]
        self._pool_to_feature_layer = None  # self.classifier[0:-1] 

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

    def feature(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def _make_layer(self, block, channels, block_num, stride=1):
        # channels: channel num of first conv in res blocks
        # downsample to adjust channel num;  adjust size by stride
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion)
            )
        layers = [block(self.in_channels, channels, downsample=downsample, stride=stride)]
        self.in_channels = channels * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)
