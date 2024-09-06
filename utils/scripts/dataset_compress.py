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
from utils.datasets import get_dataset_pkl, sample_per_class, dump_ds_pd


def compress(args):
    base_dir = os.path.join(args.dataset, args.subset) if args.base_dir is None else args.base_dir
    input_dir = os.path.join('data', base_dir, args.split) if args.input_dir is None else args.input_dir
    fname = args.split if args.fname is None else args.fname

    fname = os.path.join('data', base_dir, fname + '.pkl')
    labels = pd.read_csv(os.path.join(input_dir, 'labels.csv'))
    print("Compressing: ", input_dir, " ---> ", fname)

    ds_dump = []
    for idx, (f_cur, label_cur) in labels.iterrows():
        f_cur = Image.open(os.path.join(input_dir, "%06d.png" % f_cur))  # f_cur.strip()+'.png'
        ds_dump.append((f_cur, int(label_cur)))

    df = pd.DataFrame(ds_dump, index=None, columns=['image', 'label'])
    df.to_pickle(fname)
    print('dataset saved to ', fname)


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
    parser.add_argument('--input_dir', type=str, help='dir of datasets', default=None)  # ./data/temp
    parser.add_argument('--fname', type=str, help='train/test/val/dev', default=None)


    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    splits = ['train', 'test', 'val', 'dev'] if args.split in ['all'] else [args.split]
    for split in splits:
        args.split = split
        compress(args)

# python utils/scripts/dataset_decompress.py  --dataset cifar10 --subset clean --split train_sample_20

