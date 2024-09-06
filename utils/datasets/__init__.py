import h5py
import torch
import csv
import pathlib
import pickle
import os
import random
import copy
import functools
import numpy as np
import pandas as pd

from PIL import Image
from typing import Any, Callable, Optional, Tuple

import torchvision
from torchvision.datasets import MNIST, CIFAR10, GTSRB, VOCDetection, CIFAR100
from torch.utils.data import Dataset, IterDataPipe, MapDataPipe
import torchtext
from torchtext.data import get_tokenizer

from .offic import *


__all__ = [
    "get_dataset",
    "get_dataset_offic",
    "get_dataset_pkl",
    "get_dataset_folder",

    "get_num_classes",
    "get_img_shape",
    "sample_per_class",
    "sample_one_class",
    "pil2ndarray",
    "ndarray2pil",
    "get_indices", 
    "dump_ds_pd",
    "show_ds",

    "TorchCifar10",
    "TorchCifar100",
]


def get_dataset(path, split='train', fmt='pkl', *args, **kargs):
    if fmt in ['pkl', 'pickle', 'pklds']:
        return get_dataset_pkl(path, split, *args, **kargs)
    elif fmt in ['folder']:
        return get_dataset_folder(path, split, *args, **kargs)
    elif fmt in ['offic']:
        return get_dataset_offic(path, split, *args, **kargs)
    else:
        raise Exception("get_dataset(): fmt <{}> not supported".format(fmt))


def get_dataset_pkl(path, split='train', *args, **kargs):
    return PdDatasetCV(os.path.join('.', 'data', path), split, *args, **kargs)

def get_dataset_folder(path, split='train', *args, **kargs):
    return IFDatasetCV(os.path.join('.', 'data', path), split, *args, **kargs)


# load dataset from PyTorch API
def get_dataset_offic(dataset_name, split=None, train=True, download=True):
    # split = split if split else 'train' if train else 'test'
    assert split in ['train', 'test', 'dev', 'val', 'trainval'], "Dataset: split not support."
    train = True if split == "train" else False
    # get dataset by name
    if dataset_name in ["mnist", "MNIST"]:
        return MNIST('./data/offic', train=train, download=download)
    elif dataset_name in ["cifar", "CIFAR", "cifar10", "CIFAR10"]:
        return CIFAR10('./data/offic', train=train, download=download)
    elif dataset_name in ["gtsrb", "GTSRB"]:
        return GTSRB('./data/offic', split=split, download=download)
    elif dataset_name in ["celeba", "CelebA", 'celeba8', 'CelebA8']:
        return TorchCelebA8('./data/offic', split=split)
    elif dataset_name in ["pascal_voc", "Pascal_VOC", "VOC", "voc"]:
        return VOCDetection('./data/offic', year='2012', image_set=split, download=download)
    else:
        raise Exception("Dataset not implemented")

    

class IFDatasetCV(Dataset):
    def __init__(self, path, split, trans_ls=None, transform=None, target_transform=None, pil=True, dtype=np.float32):
        # assert split in ['train', 'test', 'dev', 'val'], 'ImageFolder CV Dataset: split not support: ' + split
        data = pd.read_csv(os.path.join(path, split, 'labels.csv')).values.tolist()
        self.data = [[os.path.join(path, split, "%06d.png" % i[0]), i[1]] for i in data]
        self.transform = transform
        self.target_transform = target_transform
        self.trans_ls = [] if trans_ls is None else trans_ls

        self.pil = pil
        self.dtype = dtype

    def __getitem__(self, idx):
        image, label = self.data[idx]
        image = Image.open(image)
        if self.pil:
            image = pil2ndarray(image, self.dtype)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        for trans in self.trans_ls:
            image, label = trans([image, label])
        return image, label

    def __len__(self):
        return len(self.data)
    
    def map(self, trans):
        self.trans_ls.append(trans)
        return self


class PdDatasetCV(Dataset):
    def __init__(self, path, split, trans_ls=None, transform=None, target_transform=None, pil=None, dtype=np.float32):
        # assert split in ['train', 'test', 'dev', 'val'], 'My CV Dataset: split not support: ' + split
        path = os.path.join(path, split + '.pkl')
        self.data = pd.read_pickle(path).values.tolist()
        self.transform = transform
        self.target_transform = target_transform
        self.trans_ls = [] if trans_ls is None else trans_ls
        if pil is None:
            pil = isinstance(self.data[0][0], Image.Image)
        self.pil = pil
        self.dtype = dtype

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.pil:
            image = pil2ndarray(image, self.dtype)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        for trans in self.trans_ls:
            image, label = trans([image, label])
        return image, label

    def __len__(self):
        return len(self.data)
    
    def map(self, trans):
        self.trans_ls.append(trans)
        return self




'''
    transforms
'''
def pil2ndarray(img, dtype=np.float32):
    img = np.atleast_3d(np.array(img, dtype=dtype) / 255)
    return img


def ndarray2pil(img):
    img = Image.fromarray((img * 255).astype(np.uint8).squeeze())
    return img


'''
    utils
'''
def get_ds_info(dataset=None):
    # return img_shape, num_classes
    if dataset.lower() in ['mnist']:
        return (1, 28, 28), 10
    elif dataset.lower() in ['cifar10', 'cifar']:
        return (3, 32, 32), 10
    elif dataset.lower() in ['gtsrb']:
        return (3, 32, 32), 43
    elif dataset.lower() in ['tiny_imagenet', 'timgnet']:
        return (3, 64, 64), 200
    elif dataset.lower() in ['tiny_imagenet_224', 'timgnet224']:
        return (3, 224, 224), 200
    elif dataset.lower() in ['celeba8']:
        return (3, 128, 128), 8
    else:
        raise Exception("Dataset info not found: ", dataset)


def get_img_shape(dataset, img_shape=None, channel=None, img_size=None, **kwargs):
    # mnist  cifar10  gtsrb  tiny_imagenet
    if img_shape is not None:
        return img_shape
    elif channel is not None and img_size is not None:
        img_shape = (channel, img_size, img_size)
    elif dataset.lower() in ['mnist']:
        img_shape = (1, 28, 28)
    elif dataset.lower() in ['cifar', 'cifar10', 'gtsrb']:
        img_shape = (3, 32, 32)
    elif dataset.lower() in ['tiny_imagenet', 'tiny_imagenet_200', 'timgnet']:
        img_shape = (3, 64, 64)
    elif dataset.lower() in ['tiny_imagenet_224', 'tiny_imagenet_200_224', 'timgnet224']:
        img_shape = (3, 224, 224)
    elif dataset.lower() in ['celeba8']:
        img_shape = (3, 128, 128)
    return img_shape


def get_num_classes(dataset, num_classes=None):
    if num_classes is not None:
        return num_classes
    elif dataset.lower() in ['mnist', 'cifar', 'cifar10']:
        num_classes = 10
    elif dataset.lower() in ['gtsrb']:
        num_classes = 43
    elif dataset.lower() in ['timgnet', 'timgnet224', 'tiny_imagenet', 'tiny_imagenet_200', 'tiny_imagenet_224', 'tiny_imagenet_200_224', ]:
        num_classes = 200
    elif dataset.lower() in ['celeba8']:
        num_classes = 8
    return num_classes


def get_cls_names(dataset, cls_names=None, num_classes=None):
    from .cls_names.cls_dict import cls_dict, cls_ls
    if cls_names is not None:
        return cls_names
    elif dataset.lower() in ['mnist']:
        cls_names = cls_ls['mnist']
    elif dataset.lower() in ['cifar', 'cifar10', 'cifar-10']:
        cls_names = cls_ls['cifar10']
    elif dataset.lower() in ['cifar100', 'cifar-100']:
        cls_names = cls_ls['cifar100']
    elif dataset.lower() in ['gtsrb']:
        cls_names = cls_ls['gtsrb']
    elif dataset.lower() in ['tiny_imagenet', 'tiny_imagenet_200', 'timgnet', 'tiny_imagenet_224', 'tiny_imagenet_200_224', 'timgnet224']:
        cls_names = cls_ls['timgnet']
    else:
        cls_names = [str(i) for i in range(num_classes)]
    return cls_names



def sample_per_class(ds, num_classes, num_per_class, label_index=1):
    ds_out = copy.copy(ds)
    data = []
    for cls_idx in range(num_classes):
        data.append(list(filter(lambda x: x[label_index] == cls_idx, ds.data))[:num_per_class])
    ds_out.data = sum(data, [])
    # ds_out.data = functools.reduce(operator.iconcat, data)
    return ds_out


def sample_one_class(ds: PdDatasetCV, cls_idx) -> PdDatasetCV:
    ds_out = copy.copy(ds)
    ds_out.data = list(filter(lambda x: x[1] == cls_idx, ds.data))
    return ds_out


def merge_ds(ds1: PdDatasetCV, ds2: PdDatasetCV) -> PdDatasetCV:
    ds_out = copy.copy(ds1)
    ds_out.data = ds1.data + ds2.data
    return ds_out


    

def get_indices(ds, class_idx=None, num_classes=None):
    if isinstance(class_idx, list):
        labels = class_idx
    elif isinstance(class_idx, int):
        labels = [class_idx]
    elif class_idx is None and num_classes is not None:
        labels = list(range(num_classes))
    else:
        raise Exception("get_indices(): class_idx and num_classes should not both be None.")
    
    indices = np.where(np.isin([ds.data[i][1] for i in range(len(ds.data))], labels))[0]
    indices = indices.tolist()
    return indices



def dump_ds_pd(ds, fpath, split, info=None, index=None, columns=None, use_ndarray2pil=True):
    import json
    if use_ndarray2pil:
        for i in range(len(ds)):
            # ds[i][0] = ndarray2pil(ds[i][0])
            ds[i] = (ndarray2pil(ds[i][0]), *ds[i][1:])

    # dump dataset
    df = pd.DataFrame(ds, index=index, columns=columns)
    df.to_pickle(os.path.join(fpath, split + '.pkl'))

    # save info
    try:
        info_orig = json.load(open(os.path.join(fpath, 'stats.json'), 'r'))
    except FileNotFoundError:
        info_orig = {}
    if info is None:
        info = {}
    info_orig.update(info)
    json.dump(info_orig, open(os.path.join(fpath, 'stats.json'), 'w'), indent=4)
    print('dataset saved to ' + fpath)


def show_ds(data, path, to_pil=True):
    # data = list[[img_ndarray, label], ...]
    for i in range(len(data)):
        if to_pil:
            img = Image.fromarray((data[i][0] * 255).astype(np.uint8).squeeze())
        else:
            img = data[i][0]
        img.save(os.path.join(path, 'show', str(i) + '_' + str(data[i][1])))
    print("Dataset images saved to ", path)


