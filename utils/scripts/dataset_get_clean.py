import argparse
import os
import sys
import json
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

print(os.getcwd())
sys.path.append(os.getcwd())
from utils.datasets import get_dataset, get_dataset_offic, get_img_shape, get_num_classes, dump_ds_pd
from utils.misc import set_random_seed


def main(args):
    set_random_seed(args.seed)
    args.img_shape = get_img_shape(args.dataset, args.img_shape, args.channel, args.img_size)
    args.num_classes = get_num_classes(args.dataset, args.num_classes)
    # set output dir
    subset = 'clean'
    output_dir = os.path.join('data', args.dataset, subset, )
    os.makedirs(output_dir, exist_ok=True)

    dump(args.dataset, args.split, output_dir, args.img_shape, args.shuffle)
    dump(args.dataset, args.split_val, output_dir, args.img_shape, args.shuffle)



def dump(dataset, split, output_dir, img_shape, shuffle=True):
    ds_offic = get_dataset_offic(dataset, split)
    idxes = list(range(len(ds_offic)))
    if args.shuffle:
        random.shuffle(idxes)
    ds_dump = []
    for idx in tqdm(idxes, file=sys.stdout):
        ds_dump.append([ds_offic[idx][0], ds_offic[idx][1]])

    info = {split: {'dataset': dataset, 'img_shape': img_shape}}
    dump_ds_pd(ds_dump, output_dir, split, info, columns=['image', 'label'], use_ndarray2pil=False)



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='global seed', default=0)
    parser.add_argument('--shuffle', action='store_true', help='gen random pos idx if true')
    parser.add_argument('--ds_fmt', type=str, help='dataset format: pickle(pkl)/folder/offic', default='pkl')
    parser.add_argument('--split', type=str, help='train/test/val/dev', default='train')
    parser.add_argument('--split_val', type=str, help='train/test/val/dev', default='test')

    # dataset & inject config
    parser.add_argument('--dataset', type=str, help='name of the dataset', default='cifar10')
    parser.add_argument('--num_classes', type=int, help='classes num of the dataset', default=None)
    parser.add_argument('--img_shape', type=list, nargs='+', default=None)
    parser.add_argument('--channel', type=int, help='channel num of imgs', default=None)
    parser.add_argument('--img_size', type=int, help='height/width of imgs', default=None)

    return parser.parse_args(argv)



if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
    # print('fin')
    print()
