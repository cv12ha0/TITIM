import warnings
import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

print(os.getcwd())
sys.path.append(os.getcwd())
from utils.datasets import get_dataset_pkl, ndarray2pil
from utils.misc import set_random_seed, get_device, get_nw, OfficPreProcess, ImageProcess


def main():
    base_dir = os.path.join(args.dataset, args.subset) if args.base_dir is None else args.base_dir
    output_dir = os.path.join('data', base_dir) if args.output_dir is None else args.output_dir
    fname = os.path.join('data', base_dir, args.split + '.pkl')
    if not os.path.exists(fname):
        print('File not exist: ', fname)
        return
    os.makedirs(output_dir, exist_ok=True)
    print('loading dataset: ', fname)
    ds = get_dataset_pkl(base_dir, args.split)

    # apply transform
    ds = ds.shuffle() if args.shuffle else ds
    ds = ds.map(div255) if args.div255 else ds
    ds = ds.map(topil) if args.topil else ds

    # dump
    ds_dump = []
    prog_bar = tqdm(ds, file=sys.stdout)
    for idx, item in enumerate(prog_bar, 0):
        ds_dump.append(item)
    df = pd.DataFrame(ds_dump, columns=['image', 'label'])
    df.to_pickle(os.path.join(output_dir, split + '.pkl'))


def div255(x):
    image, label = x
    image = image / 255
    return image, label


def topil(x):
    image, label = x
    image = ndarray2pil(image)
    return image, label


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, help='GPU id', default='0')
    parser.add_argument('--seed', type=int, help='global seed', default=0)    

    parser.add_argument('--dataset', type=str, help='name of the dataset', default='cifar10')
    parser.add_argument('--subset', type=str, help='subset of dataset', default='clean')
    parser.add_argument('--split', type=str, help='train/test/val/dev', default='all')
    parser.add_argument('--num_classes', type=int, help='classes num of the dataset', default=None)
    parser.add_argument('--img_shape', type=list, nargs='+', default=None)
    parser.add_argument('--channel', type=int, help='channel num of imgs', default=None)
    parser.add_argument('--img_size', type=int, help='height/width of imgs', default=None)

    parser.add_argument('--shuffle', action='store_true', help='shuffle dataset')
    parser.add_argument('--div255', action='store_true', help='uint8 to float')
    parser.add_argument('--topil', action='store_true', help='ndarray to PIL image')
    
    parser.add_argument('--base_dir', type=str, help='dir of datasets', default=None)
    parser.add_argument('--output_dir', type=str, help='dir of datasets', default=None)  # ./data/temp


    return parser.parse_args(argv)

if __name__ == '__main__':
    # import ssl
    # ssl._create_default_https_context = ssl._create_unverified_context
    args = parse_arguments(sys.argv[1:])
    splits = ['train', 'test', 'val', 'dev'] if args.split in ['all'] else [args.split]
    for split in splits:
        args.split = split
        main()    
