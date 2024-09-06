import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

print(os.getcwd())
sys.path.append(os.getcwd())
from utils.datasets import get_dataset_pkl, sample_per_class


def decompress(args):
    base_dir = os.path.join(args.dataset, args.subset) if args.base_dir is None else args.base_dir
    output_dir = os.path.join('data', base_dir, args.split) if args.output_dir is None else args.output_dir

    fname = os.path.join('data', base_dir, args.split + '.pkl')
    fname_csv = os.path.join(output_dir, 'labels.csv')
    if not os.path.exists(fname):
        print('File not exist: ', fname)
        return
    os.makedirs(output_dir, exist_ok=True)
    print("Decompressing: ", fname, " ---> ", output_dir)

    ds = get_dataset_pkl(base_dir, args.split)
    label_info = (('%20s,' * len(ds.data[0]) % ('name', 'label')[:(len(ds.data[0]))]).rstrip(',') + '\n')
    prog_bar = tqdm(ds.data, file=sys.stdout, leave=True, disable=False)
    for idx, data in enumerate(prog_bar):
        img = data[0]
        if isinstance(img, np.ndarray):
            if img.dtype is float:
                img = img * 255
            img = Image.fromarray(img.astype(np.uint8))
        img.save(os.path.join(output_dir, "%06d.png" % idx))

        if len(data) > 1:
            # label_info += '%20.5g,%20.5g' % (idx, data[idx][1]) + '\n'
            label_info += '{:>20},{:>20}'.format("%06d" % idx, data[1]) + '\n'

    with open(fname_csv, 'a+') as f:
        f.write(label_info)
    print('dataset saved to ', output_dir)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, help='GPU id', default='0')
    parser.add_argument('--seed', type=int, help='global seed', default=0)    
    parser.add_argument('--shuffle', action='store_true', help='gen random pos idx if true')
    parser.add_argument('--config', type=str, help='dir of config file', default='imdb_test')

    parser.add_argument('--dataset', type=str, help='name of the dataset', default='cifar10')
    parser.add_argument('--subset', type=str, help='subset of dataset', default='clean')
    parser.add_argument('--split', type=str, help='train/test/val/dev', default='all')
    parser.add_argument('--num_classes', type=int, help='classes num of the dataset', default=None)
    parser.add_argument('--img_shape', type=list, nargs='+', default=None)
    parser.add_argument('--channel', type=int, help='channel num of imgs', default=None)
    parser.add_argument('--img_size', type=int, help='height/width of imgs', default=None)
    

    parser.add_argument('--base_dir', type=str, help='dir of datasets', default=None)
    parser.add_argument('--output_dir', type=str, help='dir of datasets', default=None)  # ./data/temp


    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    splits = ['train', 'test', 'val', 'dev'] if args.split in ['all'] else [args.split]
    for split in splits:
        args.split = split
        decompress(args)

# python utils/scripts/dataset_decompress.py  --dataset cifar10 --subset clean --split train_sample_20

