import numpy as np
import torch
import torchvision
from torch import nn


class AlexNet(nn.Module):
    """
    Input_shape should be 3*227*227 (3*224*224 in paper), and should run parallel on 2 GPUs
    """
    def __init__(self, img_shape=(3, 224, 224), num_classes=1000):
        super().__init__()
        self.c, self.h, self.w = img_shape
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.config = {'model': 'AlexNet', 'img_shape': self.img_shape, 'num_classes': self.num_classes, }

        self.features = nn.Sequential(
            nn.Conv2d(self.c, 48*2, kernel_size=11, stride=4, padding=2),  # pad to match input shape in paper
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48*2, 128*2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128*2, 192*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192*2, 192*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192*2, 128*2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 2 * 128 * 6 * 6, 9216
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x


class AlexNetMod(nn.Module):
    """
    Modified AlexNet
    From: https://github.com/icpm/pytorch-cifar10/blob/master/models/AlexNet.py
    """
    def __init__(self, img_shape=(3, 224, 224), num_classes=10):
        super().__init__()
        self.c, self.h, self.w = img_shape
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.config = {'model': 'AlexNet', 'img_shape': self.img_shape, 'num_classes': self.num_classes, }

        self.features = nn.Sequential(
            nn.Conv2d(self.c, 64, kernel_size=3, stride=2, padding=1),
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
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x
