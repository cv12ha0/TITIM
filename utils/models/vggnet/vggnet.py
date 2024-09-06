"""
    VGG11/13/16/19 in Pytorch.
"""
import torch
import torch.nn as nn


def vggnet11(img_shape, num_classes=1000, init_weighgt=True):
    # pretrained weights:  'https://download.pytorch.org/models/vgg11-bbd30ac9.pth'
    model = VGGNet('vgg11', img_shape, num_classes, init_weighgt)
    model.config.update({'model': 'VGGNet11', })
    return model


def vggnet13(img_shape, num_classes=1000, init_weighgt=True):
    # pretrained weights:  'https://download.pytorch.org/models/vgg13-c768596a.pth'
    model = VGGNet('vgg13', img_shape, num_classes, init_weighgt)
    model.config.update({'model': 'VGGNet13', })
    return model


def vggnet16(img_shape, num_classes=1000, init_weighgt=True):
    # pretrained weights:  'https://download.pytorch.org/models/vgg16-397923af.pth'
    model = VGGNet('vgg16', img_shape, num_classes, init_weighgt)
    model.config.update({'model': 'VGGNet16', })
    return model


def vggnet19(img_shape, num_classes=1000, init_weighgt=True):
    # pretrained weights:  'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
    model = VGGNet('vgg19', img_shape, num_classes, init_weighgt)
    model.config.update({'model': 'VGGNet19', })
    return model


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGGNet(nn.Module):

    def __init__(self, cfg_name='vgg16', img_shape=(3, 224, 224), num_classes=10, init_weights=True):
        super().__init__()
        self.c, self.h, self.w = img_shape
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.config = {'model': 'VGGNet', 'img_shape': self.img_shape, 'num_classes': self.num_classes, }

        self.features = self._make_layers(cfgs[cfg_name])
        self.classifier = nn.Sequential(
            nn.Linear(int(512 * self.h / 32 * self.w / 32), 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1000)
        )
        self.fc = nn.Linear(1000, num_classes)
        # self.classifier = nn.Linear(512, 10)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.fc(x)
        return x

    def feature(self, x):
        x = self.features[:-1](x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.c
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
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


if __name__ == '__main__':
    net = VGGNet('VGG11')
    input_temp = torch.randn(2, 3, 32, 32)
    out = net(input_temp)
    print(out.size())
