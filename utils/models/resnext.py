'''ResNeXt in PyTorch.

See the paper "Aggregated Residual Transformations for Deep Neural Networks" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


def resnext29_2x64d(img_shape, num_classes=1000, ):
    model = ResNeXt(ResNeXtBottleneck, [3, 3, 3, 0], 2, 64, img_shape, num_classes)
    model.config.update({'model': 'ResNext29_2x64D', })
    return model


def resnext29_4x64d(img_shape, num_classes=1000, ):
    model = ResNeXt(ResNeXtBottleneck, [3, 3, 3, 0], 4, 64, img_shape, num_classes)
    model.config.update({'model': 'ResNext29_4x64D', })
    return model


def resnext29_8x64d(img_shape, num_classes=1000, ):
    model = ResNeXt(ResNeXtBottleneck, [3, 3, 3, 0], 8, 64, img_shape, num_classes)
    model.config.update({'model': 'ResNext29_8x64D', })
    return model


def resnext29_32x4d(img_shape, num_classes=1000, ):
    model = ResNeXt(ResNeXtBottleneck, [3, 3, 3, 0], 32, 4, img_shape, num_classes)
    model.config.update({'model': 'ResNext29_32x4D', })
    return model
    

def resnext47_8x64d(img_shape, num_classes=1000, ):
    model = ResNeXt(ResNeXtBottleneck, [5, 5, 5, 0], 8, 64, img_shape, num_classes)
    model.config.update({'model': 'ResNext47_8x64D', })
    return model


def resnext47_16x64d(img_shape, num_classes=1000, ):
    model = ResNeXt(ResNeXtBottleneck, [5, 5, 5, 0], 16, 64, img_shape, num_classes)
    model.config.update({'model': 'ResNext47_16x64D', })
    return model


def resnext47_32x4d(img_shape, num_classes=1000, ):
    model = ResNeXt(ResNeXtBottleneck, [5, 5, 5, 0], 32, 4, img_shape, num_classes)
    model.config.update({'model': 'ResNext47_32x4D', })
    return model


def resnext50_32x4d(img_shape, num_classes=1000, ):
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], 32, 4, img_shape, num_classes)
    model.config.update({'model': 'ResNext50_32x4D', })
    return model


def resnext101_32x4d(img_shape, num_classes=1000, ):
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], 32, 4, img_shape, num_classes)
    model.config.update({'model': 'ResNext101_32x4D', })
    return model


def resnext152_32x4d(img_shape, num_classes=1000, ):
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 36, 3], 32, 4, img_shape, num_classes)
    model.config.update({'model': 'ResNext152_32x4D', })
    return model




class ResNeXtBlock(nn.Module):
    '''<Deprecated>   Grouped convolution block.'''
    expansion = 2

    def __init__(self, in_planes, out_planes, cardinality=32, bottleneck_width=4, stride=1, downsample=None):
        super().__init__()
        group_width = cardinality * bottleneck_width
        width = int(out_planes * (bottleneck_width / 64.0)) * cardinality
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, self.expansion*group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*group_width)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*group_width:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*group_width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*group_width)
            )
        

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNeXtBottleneck(nn.Module):
    """
    Basic block in ResNet50, 101, 152, ...
    """
    expansion = 4

    def __init__(self, in_planes, out_planes, cardinality, bottleneck_width, stride=1, downsample=None):
        super().__init__()
        width = int(out_planes * (bottleneck_width / 64.0)) * cardinality
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes * self.expansion)
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


class ResNeXt(nn.Module):
    def __init__(self, block, num_blocks, cardinality, bottleneck_width, img_shape=(3, 224, 224), num_classes=1000):
        super().__init__()
        self.c, self.h, self.w = img_shape
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64
        self.config = {'model': 'ResNeXt', 'img_shape': self.img_shape, 'num_classes': self.num_classes, 
                       'cardinality': cardinality, 'bottleneck_width': bottleneck_width}


        self.conv1 = nn.Conv2d(self.c, self.in_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2)
        if num_blocks[3] != 0:
            self.layer4 = self._make_layer(block, 512, num_blocks[3], 2)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.layer4 = nn.Sequential()
            self.fc = nn.Linear(256 * block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.linear = nn.Linear(cardinality*bottleneck_width*8, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None  # nn.Sequential()
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        
        layers = [block(self.in_planes, planes, self.cardinality, self.bottleneck_width, downsample=downsample, stride=stride)]
        self.in_planes = planes * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, self.cardinality, self.bottleneck_width))

        return nn.Sequential(*layers)



    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out




