import numpy as np
import torch
import torchvision
import torchvision.transforms
from torch import nn

__all__ = [
    "get_model",

    "freeze",
    "freeze_after",
    "show_freeze_status"

]
# classification
from .lenet import *
from .alexnet import *
from .vggnet import *
from .googlenet import *
from .xception import *
from .resnet import *
from .wrn import * 
from .preact_resnet import *
from .resnext import *
from .densenet import *
from .dla import *
from .dpn import *
from .rir import *
from .senet import *
from .mobilenet import *
from .mobilenetv2 import *
from .efficientnet import *
from .regnet import *
from .nasnet import *
from .pnasnet import *
from .shufflenet import *
from .shufflenetv2 import *
from .squeezenet import *
from .resatt import *

# generate
from .unet import UNet


def get_model(name, img_shape, num_classes=10, device='cpu'):
    if name in ['mnist', 'mnistmodel']:
        model = MNISTModel()

    # LeNet
    elif name in ['lenet', 'LeNet', 'LeNet5']:
        model = LeNet5(img_shape, num_classes)

    # AlexNet
    elif name in ['alexnet', 'AlexNet']:
        model = AlexNet(img_shape, num_classes)
    elif name in ['alexnetmod', 'AlexNetMod']:
        model = AlexNetMod(img_shape, num_classes)

    # GoogLeNet / Inception
    elif name in ['googlenet', 'GoogLeNet']:
        model = googlenet(img_shape, num_classes)
    elif name in ['inceptionv3', 'InceptionV3']:
        model = inception_v3(img_shape, num_classes, aux_logits=True, transform_input=False)
    elif name in ['inceptionv4', 'InceptionV4']:
        model = inception_v4(img_shape, num_classes)
    elif name in ['inceptionresnetv2', 'InceptionResNetV2']:
        model = inception_resnet_v2(img_shape, num_classes)
    
    # Xception
    elif name in ['xception', 'Xception']:
        model = xception(img_shape, num_classes)

    # VGGNet
    elif name in ['vgg11', 'VGG11', 'vggnet11', 'VGGNet11']:
        model = vggnet11(img_shape, num_classes, True)
    elif name in ['vgg13', 'VGG13', 'vggnet13', 'VGGNet13']:
        model = vggnet13(img_shape, num_classes, True)
    elif name in ['vgg', 'VGG', 'vgg16', 'VGG16', 'vggnet', 'VGGNet', 'vggnet16', 'VGGNet16']:
        model = vggnet16(img_shape, num_classes, True)
    elif name in ['vgg19', 'VGG19', 'vggnet19', 'VGGNet19']:
        model = vggnet19(img_shape, num_classes, True)
    # VGGNet backbone ver
    elif name in ['vgg11_backbone', 'VGG11_backbone', 'vggnet11_backbone', 'VGGNet11_backbone']:
        model = vggnet11(img_shape, num_classes, True)
    elif name in ['vgg13_backbone', 'VGG13_backbone', 'vggnet13_backbone', 'VGGNet13_backbone']:
        model = vggnet13(img_shape, num_classes, True)
    elif name in ['vgg_backbone', 'VGG_backbone', 'vgg16_backbone', 'VGG16_backbone', 'vggnet_backbone', 'VGGNet_backbone', 'vggnet16_backbone', 'VGGNet16_backbone']:
        model = vggnet16(img_shape, num_classes, True)
    elif name in ['vgg19_backbone', 'VGG19_backbone', 'vggnet19_backbone', 'VGGNet19_backbone']:
        model = vggnet19(img_shape, num_classes, True)

    # ResNet
    elif name in ['resnet', 'ResNet', 'resnet18', 'ResNet18']:
        model = resnet18(img_shape, num_classes, True)
    elif name in ['resnet34', 'ResNet34']:
        model = resnet34(img_shape, num_classes, True)
    elif name in ['resnet50', 'ResNet50']:
        model = resnet50(img_shape, num_classes, True)
    elif name in ['resnet101', 'ResNet101']:
        model = resnet101(img_shape, num_classes, True)
    elif name in ['resnet152', 'ResNet152']:
        model = resnet152(img_shape, num_classes, True)
    
    # WideResNet / WRN
    elif name in ['wrn', 'WRN', 'wideresnet', 'WideResNet']:
        model = wide_resnet(img_shape, num_classes, depth=40, widen_factor=10, )

    # PreActResNet
    elif name in ['preactresnet', 'PreActResNet', 'preactresnet18', 'PreActResNet18']:
        model = preactresnet18(img_shape, num_classes, )
    elif name in ['preactresnet34', 'PreActResNet34']:
        model = preactresnet34(img_shape, num_classes, )
    elif name in ['preactresnet50', 'PreActResNet50']:
        model = preactresnet50(img_shape, num_classes, )
    elif name in ['preactresnet101', 'PreActResNet101']:
        model = preactresnet101(img_shape, num_classes, )
    elif name in ['preactresnet152', 'PreActResNet152']:
        model = preactresnet152(img_shape, num_classes, )

    # ResNeXt
    elif name in ['resnext', 'ResNeXt', 'resnext29_2x64d', 'ResNeXt29_2x64D']:
        model = resnext29_2x64d(img_shape, num_classes)
    elif name in ['resnext29_4x64d', 'ResNeXt29_4x64D']:
        model = resnext29_4x64d(img_shape, num_classes)
    elif name in ['resnext29_8x64d', 'ResNeXt29_8x64D']:
        model = resnext29_8x64d(img_shape, num_classes)
    elif name in ['resnext29_32x4d', 'ResNeXt29_32x4D']:
        model = resnext29_32x4d(img_shape, num_classes)
    elif name in ['resnext47_8x64d', 'ResNeXt47_8x64D']:
        model = resnext47_8x64d(img_shape, num_classes)
    elif name in ['resnext47_16x64d', 'ResNeXt47_16x64D']:
        model = resnext47_16x64d(img_shape, num_classes)
    elif name in ['resnext47_32x4d', 'ResNeXt47_32x4D']:
        model = resnext47_32x4d(img_shape, num_classes)

    elif name in ['resnext50', 'ResNeXt50', 'resnext50_32x4d', 'ResNeXt50_32x4D']:
        model = resnext50_32x4d(img_shape, num_classes)
    elif name in ['resnext101', 'ResNeXt101', 'resnext101_32x4d', 'ResNeXt101_32x4D']:
        model = resnext101_32x4d(img_shape, num_classes)
    elif name in ['resnext152', 'ResNeXt152', 'resnext152_32x4d', 'ResNeXt152_32x4D']:
        model = resnext152_32x4d(img_shape, num_classes)

    # DLA
    elif name in ['dla', 'DLA']:
        model = dla(img_shape, num_classes)

    # DLA
    elif name in ['dpn26', 'DPN26']:
        model = dpn26(img_shape, num_classes)
    elif name in ['dpn92', 'DPN92']:
        model = dpn92(img_shape, num_classes)

    # RIR
    elif name in ['rir', 'RIR', 'resnetinresneet', 'ResnetInResneet']:
        model = rir(img_shape, num_classes)
    
    # SENet
    elif name in ['senet', 'SENet', 'senet18', 'SENet18']:
        model = senet18(img_shape, num_classes)
    elif name in ['senet34', 'SENet34']:
        model = senet34(img_shape, num_classes)
    elif name in ['senet50', 'SENet50']:
        model = senet50(img_shape, num_classes)
    elif name in ['senet101', 'SENet101']:
        model = senet101(img_shape, num_classes)
    elif name in ['senet152', 'SENet152']:
        model = senet152(img_shape, num_classes)


    # MobileNet
    elif name in ['mobilenet', 'MobileNet']:
        model = mobilenet(img_shape, num_classes)
    elif name in ['mobilenetv2', 'MobileNetV2']:
        model = mobilenetv2(img_shape, num_classes)

    # EfficientNet
    elif name in ['efficientnet', 'EfficientNet', 'efficientnetb0', 'EfficientNetB0']:
        model = efficientnetb0(img_shape, num_classes)
    
    # RegNet
    elif name in ['regnet', 'RegNet', 'regnetx200mf', 'RegNetX_200MF']:
        model = regnetx200mf(img_shape, num_classes)
    elif name in ['regnetx400mf', 'RegNetX_400MF']:
        model = regnetx400mf(img_shape, num_classes)
    elif name in ['regnety400mf', 'RegNetY_400MF']:
        model = regnety400mf(img_shape, num_classes)

    # NasNet
    elif name in ['nasneta', 'NasNetA']:
        model = nasneta(img_shape, num_classes)

    # PNASNet
    elif name in ['pnasneta', 'PNASNetA']:
        model = pnasneta(img_shape, num_classes)
    elif name in ['pnasnetb', 'PNASNetB']:
        model = pnasnetb(img_shape, num_classes)

    # ShuffleNet
    elif name in ['shufflenetg1', 'ShuffleNetG1']:
        model = shufflenetg1(img_shape, num_classes)
    elif name in ['shufflenet', 'ShuffleNet', 'shufflenetg2', 'ShuffleNetG2']:
        model = shufflenetg2(img_shape, num_classes)
    elif name in ['shufflenetg3', 'ShuffleNetG3']:
        model = shufflenetg3(img_shape, num_classes)

    # ShuffleNetV2
    elif name in ['shufflenetv2x0.5', 'ShuffleNetV2X0.5']:
        model = shufflenetv2(img_shape, num_classes, 0.5)
    elif name in ['shufflenetv2', 'ShuffleNetV2', 'shufflenetv2x1', 'ShuffleNetV2X1']:
        model = shufflenetv2(img_shape, num_classes, 1)
    elif name in ['shufflenetv2x1.5', 'ShuffleNetV2X1.5']:
        model = shufflenetv2(img_shape, num_classes, 1.5)
    elif name in ['shufflenetv2x2', 'ShuffleNetV2X2']:
        model = shufflenetv2(img_shape, num_classes, 2)

    # SqueezeNet
    elif name in ['squeezenet', 'SqueezeNet']:
        model = squeezenet(img_shape, num_classes)

    # DenseNet
    elif name in ['densenet', 'DenseNet', 'densenet121', 'DenseNet121']:
        model = densenet121(img_shape, num_classes)
    elif name in ['densenet169', 'DenseNet169']:
        model = densenet169(img_shape, num_classes)
    elif name in ['densenet201', 'DenseNet201']:
        model = densenet201(img_shape, num_classes)
    elif name in ['densenet264', 'DenseNet264']:
        model = densenet264(img_shape, num_classes)
    elif name in ['densenet161', 'DenseNet161']:
        model = densenet161(img_shape, num_classes)
    elif name in ['densenetcifar', 'DenseNetcifar', 'densenet_cifar', 'DenseNet_cifar']:
        model = densenet_cifar(img_shape, num_classes)

    # Residual Attention Network
    elif name in ['resatt56', 'ResAtt56', 'resatt', 'ResAtt']:
        model = resatt56(img_shape, num_classes)
    elif name in ['resatt92', 'ResAtt92']:
        model = resatt92(img_shape, num_classes)

    else:
        raise Exception("Model not implemented")
    return model.to(device)


'''
    misc models
'''
class MNISTModel(nn.Module):
    def __init__(self, init_weights=True):
        super(MNISTModel, self).__init__()
        self.img_shape = (1, 28, 28)
        self.num_classes = 10
        self.c, self.h, self.w = self.img_shape
        self.config = {'model': 'mnistmodel', 'img_shape': self.img_shape, 'num_classes': self.num_classes,
                       'init_weights': init_weights}

        self.conv = nn.Sequential(
            nn.Conv2d(self.c, 16, (5, 5), padding='same'),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, (5, 5), padding='same'),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1568, 512),
            nn.ReLU(True),
            nn.Linear(512, self.num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        # print(x.shape)
        x = self.fc(x)
        return x
    
    def feature(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



class TorchResNet18(torch.nn.Module):
    def __init__(self, img_shape, num_classes=10):
        super().__init__()
        from torchvision.models import resnet18
        self.resnet = resnet18(num_classes=num_classes)
        self.img_shape = img_shape
        self.c, self.h, self.w = self.img_shape
        self.num_classes = num_classes
        self.config = {'model': 'TorchResNet18', 'img_shape': self.img_shape, 'num_classes': self.num_classes}

    def forward(self, x):
        x = self.resnet(x)
        return x




'''
    utils
'''
def freeze(model, ):
    for layer, param in model.named_parameters():
        param.requires_grad = False
    return model


def freeze_after(model, layer_name='classifier.6'):
    layer_name += '.weight'
    flag = False
    for layer, param in model.named_parameters():
        if layer == layer_name:
            flag = True
        if flag:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

def show_freeze_status(model, detail=False):
    num_freezed, num_pass = 0, 0
    print("Model Status: ")
    for (name, param) in model.named_parameters():
        is_freezed = param.requires_grad
        if is_freezed:
            num_freezed += 1
        else:
            num_pass += 1
        if detail:
            print("   ", name, " : freezed" if is_freezed else "  : ")
    print("Total params: {}  freezed: {}".format(num_freezed + num_pass, num_freezed))
























