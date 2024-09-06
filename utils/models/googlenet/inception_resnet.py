'''
    Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning (https://arxiv.org/abs/1602.07261)

    code:
        https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/inceptionv4.py
'''
import torch
import torch.nn as nn

__all__ = ['InceptionResNetV2', 'inception_resnet_v2']


def inception_resnet_v2(img_shape, num_classes=1000):
    model = InceptionResNetV2(5, 10, 5, img_shape=img_shape, num_classes=num_classes)
    model.config.update({'model': 'InceptionResNetV2', })
    return model


class BasicConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Inception_Stem(nn.Module):
    #"""Figure 3. The schema for stem of the pure Inception-v4 and
    #Inception-ResNet-v2 networks. This is the input part of those
    #networks."""
    def __init__(self, input_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            BasicConv2d(input_channels, 32, kernel_size=3),
            BasicConv2d(32, 32, kernel_size=3, padding=1),
            BasicConv2d(32, 64, kernel_size=3, padding=1)
        )

        self.branch3x3_conv = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3_pool = nn.MaxPool2d(3, stride=1, padding=1)

        self.branch7x7a = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1),
            BasicConv2d(64, 64, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(64, 64, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(64, 96, kernel_size=3, padding=1)
        )

        self.branch7x7b = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1)
        )

        self.branchpoola = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branchpoolb = BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = [self.branch3x3_conv(x), self.branch3x3_pool(x)]
        x = torch.cat(x, 1)

        x = [self.branch7x7a(x), self.branch7x7b(x)]
        x = torch.cat(x, 1)

        x = [self.branchpoola(x), self.branchpoolb(x)]
        x = torch.cat(x, 1)

        return x


class InceptionResNetA(nn.Module):
    #"""Figure 16. The schema for 35 × 35 grid (Inception-ResNet-A)
    #module of the Inception-ResNet-v2 network."""
    def __init__(self, input_channels):
        super().__init__()
        self.branch3x3stack = nn.Sequential(
            BasicConv2d(input_channels, 32, kernel_size=1),
            BasicConv2d(32, 48, kernel_size=3, padding=1),
            BasicConv2d(48, 64, kernel_size=3, padding=1)
        )

        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels, 32, kernel_size=1),
            BasicConv2d(32, 32, kernel_size=3, padding=1)
        )

        self.branch1x1 = BasicConv2d(input_channels, 32, kernel_size=1)

        self.reduction1x1 = nn.Conv2d(128, 384, kernel_size=1)
        self.shortcut = nn.Conv2d(input_channels, 384, kernel_size=1)
        self.bn = nn.BatchNorm2d(384)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = [
            self.branch1x1(x),
            self.branch3x3(x),
            self.branch3x3stack(x)
        ]

        residual = torch.cat(residual, 1)
        residual = self.reduction1x1(residual)
        shortcut = self.shortcut(x)

        output = self.bn(shortcut + residual)
        output = self.relu(output)

        return output


class InceptionResNetB(nn.Module):
    #"""Figure 17. The schema for 17 × 17 grid (Inception-ResNet-B) module of
    #the Inception-ResNet-v2 network."""
    def __init__(self, input_channels):
        super().__init__()
        self.branch7x7 = nn.Sequential(
            BasicConv2d(input_channels, 128, kernel_size=1),
            BasicConv2d(128, 160, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(160, 192, kernel_size=(7, 1), padding=(3, 0))
        )

        self.branch1x1 = BasicConv2d(input_channels, 192, kernel_size=1)

        self.reduction1x1 = nn.Conv2d(384, 1154, kernel_size=1)
        self.shortcut = nn.Conv2d(input_channels, 1154, kernel_size=1)

        self.bn = nn.BatchNorm2d(1154)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = [self.branch1x1(x), self.branch7x7(x)]
        residual = torch.cat(residual, 1)

        #"""In general we picked some scaling factors between 0.1 and 0.3 to scale the residuals
        #before their being added to the accumulated layer activations (cf. Figure 20)."""
        residual = self.reduction1x1(residual) * 0.1
        shortcut = self.shortcut(x)

        output = self.bn(residual + shortcut)
        output = self.relu(output)
        return output


class InceptionResNetC(nn.Module):
    def __init__(self, input_channels):
        #Figure 19. The schema for 8×8 grid (Inception-ResNet-C)
        #module of the Inception-ResNet-v2 network."""
        super().__init__()
        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels, 192, kernel_size=1),
            BasicConv2d(192, 224, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(224, 256, kernel_size=(3, 1), padding=(1, 0))
        )

        self.branch1x1 = BasicConv2d(input_channels, 192, kernel_size=1)
        self.reduction1x1 = nn.Conv2d(448, 2048, kernel_size=1)
        self.shorcut = nn.Conv2d(input_channels, 2048, kernel_size=1)
        self.bn = nn.BatchNorm2d(2048)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = [self.branch1x1(x), self.branch3x3(x)]
        residual = torch.cat(residual, 1)
        residual = self.reduction1x1(residual) * 0.1
        shorcut = self.shorcut(x)

        output = self.bn(shorcut + residual)
        output = self.relu(output)
        return output


class InceptionResNetReductionA(nn.Module):
    #"""Figure 7. The schema for 35 × 35 to 17 × 17 reduction module.
    #Different variants of this blocks (with various number of filters)
    #are used in Figure 9, and 15 in each of the new Inception(-v4, - ResNet-v1,
    #-ResNet-v2) variants presented in this paper. The k, l, m, n numbers
    #represent filter bank sizes which can be looked up in Table 1.
    def __init__(self, input_channels, k, l, m, n):
        super().__init__()
        self.branch3x3stack = nn.Sequential(
            BasicConv2d(input_channels, k, kernel_size=1),
            BasicConv2d(k, l, kernel_size=3, padding=1),
            BasicConv2d(l, m, kernel_size=3, stride=2)
        )

        self.branch3x3 = BasicConv2d(input_channels, n, kernel_size=3, stride=2)
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.output_channels = input_channels + n + m

    def forward(self, x):
        x = [
            self.branch3x3stack(x),
            self.branch3x3(x),
            self.branchpool(x)
        ]

        return torch.cat(x, 1)


class InceptionResNetReductionB(nn.Module):
    #"""Figure 18. The schema for 17 × 17 to 8 × 8 grid-reduction module.
    #Reduction-B module used by the wider Inception-ResNet-v1 network in
    #Figure 15."""
    #I believe it was a typo(Inception-ResNet-v1 should be Inception-ResNet-v2)
    def __init__(self, input_channels):
        super().__init__()
        self.branchpool = nn.MaxPool2d(3, stride=2)

        self.branch3x3a = nn.Sequential(
            BasicConv2d(input_channels, 256, kernel_size=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch3x3b = nn.Sequential(
            BasicConv2d(input_channels, 256, kernel_size=1),
            BasicConv2d(256, 288, kernel_size=3, stride=2)
        )

        self.branch3x3stack = nn.Sequential(
            BasicConv2d(input_channels, 256, kernel_size=1),
            BasicConv2d(256, 288, kernel_size=3, padding=1),
            BasicConv2d(288, 320, kernel_size=3, stride=2)
        )

    def forward(self, x):
        x = [
            self.branch3x3a(x),
            self.branch3x3b(x),
            self.branch3x3stack(x),
            self.branchpool(x)
        ]

        x = torch.cat(x, 1)
        return x


class InceptionResNetV2(nn.Module):
    def __init__(self, A, B, C, k=256, l=256, m=384, n=384, img_shape=(3, 224, 224), num_classes=1000):
        super().__init__()
        self.c, self.h, self.w = img_shape
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.config = {'model': 'InceptionV4', 'A': A, 'B': B, 'C': C, 'img_shape': self.img_shape, 'num_classes': self.num_classes, }

        self.stem = Inception_Stem(self.c)
        self.inception_resnet_a = self._generate_inception_module(384, 384, A, InceptionResNetA)
        self.reduction_a = InceptionResNetReductionA(384, k, l, m, n)
        output_channels = self.reduction_a.output_channels
        self.inception_resnet_b = self._generate_inception_module(output_channels, 1154, B, InceptionResNetB)
        self.reduction_b = InceptionResNetReductionB(1154)
        self.inception_resnet_c = self._generate_inception_module(2146, 2048, C, InceptionResNetC)

        #6x6 featuresize
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #"""Dropout (keep 0.8)"""
        self.dropout = nn.Dropout2d(1 - 0.8)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_resnet_a(x)
        x = self.reduction_a(x)
        x = self.inception_resnet_b(x)
        x = self.reduction_b(x)
        x = self.inception_resnet_c(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(-1, 2048)
        x = self.fc(x)

        return x

    @staticmethod
    def _generate_inception_module(input_channels, output_channels, block_num, block):
        layers = nn.Sequential()
        for l in range(block_num):
            layers.add_module("{}_{}".format(block.__name__, l), block(input_channels))
            input_channels = output_channels

        return layers
    
