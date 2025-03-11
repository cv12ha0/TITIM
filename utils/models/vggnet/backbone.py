import numpy as np
import torch
import torchvision
from torch import nn


def vggnet11_backbone(img_shape, num_classes=1000, init_weighgt=True):
    # pretrained weights:  'https://download.pytorch.org/models/vgg11-bbd30ac9.pth'
    model = VGGNet('vgg11', img_shape, num_classes, init_weighgt)
    model.config.update({'model': 'VGGNet11', })
    return model


def vggnet13_backbone(img_shape, num_classes=1000, init_weighgt=True):
    # pretrained weights:  'https://download.pytorch.org/models/vgg13-c768596a.pth'
    model = VGGNet('vgg13', img_shape, num_classes, init_weighgt)
    model.config.update({'model': 'VGGNet13', })
    return model


def vggnet16_backbone(img_shape, num_classes=1000, init_weighgt=True):
    # pretrained weights:  'https://download.pytorch.org/models/vgg16-397923af.pth'
    model = VGGNet('vgg16', img_shape, num_classes, init_weighgt)
    model.config.update({'model': 'VGGNet16', })
    return model


def vggnet19_backbone(img_shape, num_classes=1000, init_weighgt=True):
    # pretrained weights:  'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
    model = VGGNet('vgg19', img_shape, num_classes, init_weighgt)
    model.config.update({'model': 'VGGNet19', })
    return model


class VGGNet(nn.Module):
    """
    VGGNet backbone for rcnn ...
    """
    cfgs = {
        'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

    def __init__(self, cfg_name='vgg11', img_shape=(3, 224, 224), num_classes=10, init_weights=True):
        super().__init__()
        self.c, self.h, self.w = img_shape
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.config = {'model': 'VGGNet', 'img_shape': self.img_shape, 'num_classes': self.num_classes, }

        self.features = self._make_features(self.cfgs[cfg_name])
        self.classifier = nn.Sequential(
            nn.Linear(int(512 * self.h/32 * self.w/32), 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1000)
        )

        self.fc = nn.Linear(1000, num_classes)
        if init_weights:
            self._initialize_weights()

        self.feature_pixels = 16
        self.feature_map_channels = 512
        self.feature_vector_size = 4096
        self._feature_extractor = None  # self.features[0:-1]
        self._pool_to_feature_layer = None  # self.classifier[0:-1]  

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.fc(x)
        return x

    def _make_features(self, cfg):
        layers = []
        in_channels = self.c
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def feature_extractor(self, x):
        x = self._feature_extractor(x)
        return x

    def pool_to_feature_vector(self, rois):
        rois = rois.reshape((rois.shape[0], 512 * 7 * 7))  # torch.flatten(x, 1)
        return self._pool_to_feature_layer(rois)

    def set_feature_exractor(self):
        self._feature_extractor = self.features[0:-1]
        self._pool_to_feature_layer = self.classifier[0:-1] 
