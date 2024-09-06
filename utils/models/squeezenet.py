'''
    squeezenet: Learning both Weights and Connections for Efficient Neural Networks (https://arxiv.org/abs/1506.02626)

    code:
        https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/squeezenet.py
'''

import torch
from torch import nn

__all__ = [
    "SqueezeNet", 
    "squeezenet", 
]


def squeezenet(img_shape, num_classes=1000):
    model = SqueezeNet(img_shape, num_classes, )
    model.config.update({'model': 'SqueezeNet', })
    return model


class Fire(nn.Module):
    def __init__(self, in_channel, out_channel, squzee_channel):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channel, squzee_channel, 1),
            nn.BatchNorm2d(squzee_channel),
            nn.ReLU(inplace=True)
        )
        self.expand_1x1 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 1),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )
        self.expand_3x3 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 3, padding=1),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.squeeze(x)
        x = torch.cat([self.expand_1x1(x), self.expand_3x3(x)], 1)
        return x
    

class SqueezeNet(nn.Module):
    """mobile net with simple bypass"""
    def __init__(self, img_shape=(3, 224, 224), num_classes=1000):
        super().__init__()
        self.c, self.h, self.w = img_shape
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.config = {'model': 'SqueezeNet', 'img_shape': self.img_shape, 'num_classes': self.num_classes, }

        self.stem = nn.Sequential(
            nn.Conv2d(self.c, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.fire2 = Fire(96, 128, 16)
        self.fire3 = Fire(128, 128, 16)
        self.fire4 = Fire(128, 256, 32)
        self.fire5 = Fire(256, 256, 32)
        self.fire6 = Fire(256, 384, 48)
        self.fire7 = Fire(384, 384, 48)
        self.fire8 = Fire(384, 512, 64)
        self.fire9 = Fire(512, 512, 64)

        self.conv10 = nn.Conv2d(512, num_classes, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.stem(x)

        f2 = self.fire2(x)
        f3 = self.fire3(f2) + f2
        f4 = self.fire4(f3)
        f4 = self.maxpool(f4)

        f5 = self.fire5(f4) + f4
        f6 = self.fire6(f5)
        f7 = self.fire7(f6) + f6
        f8 = self.fire8(f7)
        f8 = self.maxpool(f8)

        f9 = self.fire9(f8)
        c10 = self.conv10(f9)

        x = self.avgpool(c10)
        x = x.view(x.size(0), -1)
        return x
