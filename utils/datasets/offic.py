'''Torch API Datsets'''
import os
import random

import torch
import torchvision
from torchvision.datasets import MNIST, CIFAR10, GTSRB, VOCDetection, CIFAR100
from torch.utils.data import Dataset


__all__ = [
    # "TorchDataset",
    "TorchCifar10",
    "TorchCelebA",
    "TorchCelebA8",
    "TorchImageNet",
]

class TorchDataset(Dataset):
    def __init__(self):
        return NotImplementedError("Why?")
    
    def map(self, trans):
        self.trans_ls.append(trans)



class TorchFakeData(TorchDataset):
    def __init__(self, size=1000, img_shape=(3, 224, 224), num_classes=10, trans_ls=None, transform=None, target_transform=None):
        self.data = torchvision.datasets.FakeData(size, img_shape, num_classes, transform, target_transform)
        self.transform = transform
        self.target_transform = target_transform
        self.trans_ls = [] if trans_ls is None else trans_ls

    def __getitem__(self, idx):
        image, label = self.data.__getitem__(idx)
        for trans in self.trans_ls:
            image, label = trans([image, label])
        return image, label

    def __len__(self):
        return len(self.data)

    def sample(self, num=1):
        idx = random.randint(0, len(self.data))


class TorchCifar10(TorchDataset):
    def __init__(self, root='./data/offic', train=True, trans_ls=None, transform=None, target_transform=None):
        self.data = CIFAR10(root, train, transform, target_transform, download=True)
        self.transform = transform
        self.target_transform = target_transform
        self.trans_ls = [] if trans_ls is None else trans_ls

    def __getitem__(self, idx):
        image, label = self.data.__getitem__(idx)
        # if self.transform is not None:
        #     text = self.transform(image)
        # if self.target_transform is not None:
        #     label = self.target_transform(label)
        for trans in self.trans_ls:
            image, label = trans([image, label])
        return image, label

    def __len__(self):
        return len(self.data)

    def sample(self, num=1):
        idx = random.randint(0, len(self.data))
        return self.data__getitem__(idx)
    

class TorchCelebA(TorchDataset):
    def __init__(self, root='./data/offic', split='train', target_type='identity', trans_ls=None, transform=None, target_transform=None):
        self.data = torchvision.datasets.CelebA(root, split, target_type, transform, target_transform, download=True)
        self.transform = transform
        self.target_transform = target_transform
        self.trans_ls = [] if trans_ls is None else trans_ls

    def __getitem__(self, idx):
        image, label = self.data.__getitem__(idx)
        for trans in self.trans_ls:
            image, label = trans([image, label])
        return image, label

    def __len__(self):
        return len(self.data)

    def sample(self, num=1):
        idx = random.randint(0, len(self.data))
        return self.data__getitem__(idx)
    

class TorchCelebA8(TorchDataset):
    """
        Configuration from WaNet: https://github.com/VinAIResearch/Warping-based_Backdoor_Attack-release/blob/main/utils/dataloader.py#L143
        Choose top 3 most balanced attributes(Smiling, Mouth Slightly Open, Heavy Makeup) and concatenate them to build 8 classification classes.
    """
    def __init__(self, root='./data/offic', split='train', trans_ls=None, transform=None, target_transform=None):
        self.dataset = torchvision.datasets.CelebA(root, split, target_type="attr", transform=transform, target_transform=target_transform, download=True)
        self.split = split
        self.attr_idx = [18, 31, 21]
        self.transform = transform
        self.target_transform = target_transform
        self.trans_ls = [] if trans_ls is None else trans_ls

    def _convert_attributes(self, attrs):
        return (attrs[0] << 2) + (attrs[1] << 1) + (attrs[2])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, target = self.dataset[index]
        # input = self.transform(input)
        target = self._convert_attributes(target[self.attr_idx])  # .item()
        if isinstance(target, torch.Tensor):
            target = target.item()

        for trans in self.trans_ls:
            input, target = trans([input, target])
        return (input, target)



class TorchImageNet(TorchDataset):
    """
        ImageNet 2012 classification dataset
    """
    def __init__(self, root='./data/offic', split='train', trans_ls=None, transform=None, target_transform=None):
        self.data = torchvision.datasets.ImageNet(root, split, transform, target_transform)
        self.transform = transform
        self.target_transform = target_transform
        self.trans_ls = [] if trans_ls is None else trans_ls

    def __getitem__(self, idx):
        image, label = self.data.__getitem__(idx)
        for trans in self.trans_ls:
            image, label = trans([image, label])
        return image, label

    def __len__(self):
        return len(self.data)

    def sample(self, num=1):
        idx = random.randint(0, len(self.data))
        return self.data__getitem__(idx)


