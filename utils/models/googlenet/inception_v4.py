'''
    Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning (https://arxiv.org/abs/1602.07261)

    code:
        https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/inceptionv4.py
'''
import torch
import torch.nn as nn

__all__ = ['InceptionV4', 'inception_v4']

def inception_v4(img_shape, num_classes=1000):
    model = InceptionV4(4, 7, 3, img_shape=img_shape, num_classes=num_classes)
    model.config.update({'model': 'InceptionV4', })
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


class InceptionA(nn.Module):
    #"""Figure 4. The schema for 35 × 35 grid modules of the pure
    #Inception-v4 network. This is the Inception-A block of Figure 9."""
    def __init__(self, input_channels):
        super().__init__()

        self.branch3x3stack = nn.Sequential(
            BasicConv2d(input_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, padding=1)
        )

        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1)
        )

        self.branch1x1 = BasicConv2d(input_channels, 96, kernel_size=1)

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(input_channels, 96, kernel_size=1)
        )

    def forward(self, x):
        x = [
            self.branch3x3stack(x),
            self.branch3x3(x),
            self.branch1x1(x),
            self.branchpool(x)
        ]

        return torch.cat(x, 1)


class ReductionA(nn.Module):
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


class InceptionB(nn.Module):
    #"""Figure 5. The schema for 17 × 17 grid modules of the pure Inception-v4 network.
    #This is the Inception-B block of Figure 9."""
    def __init__(self, input_channels):
        super().__init__()

        self.branch7x7stack = nn.Sequential(
            BasicConv2d(input_channels, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(192, 224, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(224, 224, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(224, 256, kernel_size=(7, 1), padding=(3, 0))
        )

        self.branch7x7 = nn.Sequential(
            BasicConv2d(input_channels, 192, kernel_size=1),
            BasicConv2d(192, 224, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(224, 256, kernel_size=(7, 1), padding=(3, 0))
        )

        self.branch1x1 = BasicConv2d(input_channels, 384, kernel_size=1)

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(input_channels, 128, kernel_size=1)
        )

    def forward(self, x):
        x = [
            self.branch1x1(x),
            self.branch7x7(x),
            self.branch7x7stack(x),
            self.branchpool(x)
        ]

        return torch.cat(x, 1)


class ReductionB(nn.Module):
    #"""Figure 8. The schema for 17 × 17 to 8 × 8 grid-reduction mod- ule.
    #This is the reduction module used by the pure Inception-v4 network in
    #Figure 9."""
    def __init__(self, input_channels):
        super().__init__()
        self.branch7x7 = nn.Sequential(
            BasicConv2d(input_channels, 256, kernel_size=1),
            BasicConv2d(256, 256, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(256, 320, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(320, 320, kernel_size=3, stride=2, padding=1)
        )

        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=3, stride=2, padding=1)
        )

        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = [
            self.branch3x3(x),
            self.branch7x7(x),
            self.branchpool(x)
        ]

        return torch.cat(x, 1)


class InceptionC(nn.Module):
    def __init__(self, input_channels):
        #"""Figure 6. The schema for 8×8 grid modules of the pure
        #Inceptionv4 network. This is the Inception-C block of Figure 9."""
        super().__init__()

        self.branch3x3stack = nn.Sequential(
            BasicConv2d(input_channels, 384, kernel_size=1),
            BasicConv2d(384, 448, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(448, 512, kernel_size=(3, 1), padding=(1, 0)),
        )
        self.branch3x3stacka = BasicConv2d(512, 256, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3stackb = BasicConv2d(512, 256, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3 = BasicConv2d(input_channels, 384, kernel_size=1)
        self.branch3x3a = BasicConv2d(384, 256, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3b = BasicConv2d(384, 256, kernel_size=(1, 3), padding=(0, 1))

        self.branch1x1 = BasicConv2d(input_channels, 256, kernel_size=1)

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(input_channels, 256, kernel_size=1)
        )

    def forward(self, x):
        branch3x3stack_output = self.branch3x3stack(x)
        branch3x3stack_output = [
            self.branch3x3stacka(branch3x3stack_output),
            self.branch3x3stackb(branch3x3stack_output)
        ]
        branch3x3stack_output = torch.cat(branch3x3stack_output, 1)

        branch3x3_output = self.branch3x3(x)
        branch3x3_output = [
            self.branch3x3a(branch3x3_output),
            self.branch3x3b(branch3x3_output)
        ]
        branch3x3_output = torch.cat(branch3x3_output, 1)

        branch1x1_output = self.branch1x1(x)

        branchpool = self.branchpool(x)

        output = [
            branch1x1_output,
            branch3x3_output,
            branch3x3stack_output,
            branchpool
        ]

        return torch.cat(output, 1)


class InceptionV4(nn.Module):
    def __init__(self, A, B, C, k=192, l=224, m=256, n=384, img_shape=(3, 224, 224), num_classes=1000):
        super().__init__()
        self.c, self.h, self.w = img_shape
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.config = {'model': 'InceptionV4', 'A': A, 'B': B, 'C': C, 'img_shape': self.img_shape, 'num_classes': self.num_classes, }

        self.stem = Inception_Stem(self.c)
        self.inception_a = self._generate_inception_module(384, 384, A, InceptionA)
        self.reduction_a = ReductionA(384, k, l, m, n)
        output_channels = self.reduction_a.output_channels
        self.inception_b = self._generate_inception_module(output_channels, 1024, B, InceptionB)
        self.reduction_b = ReductionB(1024)
        self.inception_c = self._generate_inception_module(1536, 1536, C, InceptionC)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        #"""Dropout (keep 0.8)"""
        self.dropout = nn.Dropout2d(1 - 0.8)
        self.linear = nn.Linear(1536, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_a(x)
        x = self.reduction_a(x)
        x = self.inception_b(x)
        x = self.reduction_b(x)
        x = self.inception_c(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(-1, 1536)
        x = self.linear(x)

        return x

    @staticmethod
    def _generate_inception_module(input_channels, output_channels, block_num, block):
        layers = nn.Sequential()
        for l in range(block_num):
            layers.add_module("{}_{}".format(block.__name__, l), block(input_channels))
            input_channels = output_channels

        return layers

