import os
import sys
import json
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.getcwd())
from utils.datasets import get_dataset_pkl, get_num_classes, dump_ds_pd


def shuffle():
    # set_random_seed()
    fpath = os.path.join('data', args.dataset, args.subset) if args.output_dir is None else args.output_dir
    os.makedirs(fpath, exist_ok=True)
    print("shuffling dataset: ", args.dataset, "/", args.subset, "/", args.split)
    print("output_dir: ", fpath)

    ds = get_dataset_pkl(os.path.join(args.dataset, args.subset), args.split)
    ds_dump = []
    idxes = list(range(len(ds.data)))
    random.shuffle(idxes)
    prog_bar = tqdm(idxes, file=sys.stdout)
    for idx in prog_bar:
        ds_dump.append(ds.data[idx])

    info = json.load(open(os.path.join('data', args.dataset, args.subset, 'stats.json'), 'r'))[args.split]
    dump_ds_pd(ds_dump, fpath, args.split, info, columns=['image', 'label'], use_ndarray2pil=False)
    # print('shuffled dataset saved to ' + fpath)


def unshuffle():
    # set_random_seed()
    fpath = os.path.join('data', args.dataset, args.subset) if args.output_dir is None else args.output_dir
    os.makedirs(fpath, exist_ok=True)
    ds = get_dataset_pkl(os.path.join(args.dataset, args.subset), args.split)
    print("unshuffling dataset: ", args.dataset, "/", args.subset, "/", args.split, ",  item_num: ", len(ds.data))
    print("output_dir: ", fpath)
    
    ds_dump = []
    prog_bar = tqdm(ds.data, file=sys.stdout, leave=False)
    sample_counter = 0
    for cls_idx in range(get_num_classes(args.dataset)):
        cls_counter = 0
        for item in prog_bar:
            if item[1] == cls_idx:
                ds_dump.append(item)
                cls_counter += 1
        sample_counter += cls_counter
        print('cls_idx:', cls_idx, ' finished, total num:', sample_counter, ', sample cur:', cls_counter)

    info = json.load(open(os.path.join('data', args.dataset, args.subset, 'stats.json'), 'r'))[args.split]
    dump_ds_pd(ds_dump, fpath, args.split, info, columns=['image', 'label'], use_ndarray2pil=False)
    # print('Unshuffled dataset saved to ' + fpath)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, help='GPU id', default='0')
    parser.add_argument('--seed', type=int, help='global seed', default=0)    
    parser.add_argument('--shuffle', action='store_true', help='shuffle a dataset')
    parser.add_argument('--unshuffle', action='store_true', help='unshuffle a dataset')

    parser.add_argument('--dataset', type=str, help='name of the dataset', default='cifar10')
    parser.add_argument('--subset', type=str, help='subset of dataset', default='clean')
    parser.add_argument('--split', type=str, help='train/test/val/dev', default='train')
    parser.add_argument('--num_classes', type=int, help='classes num of the dataset', default=None)
    parser.add_argument('--img_shape', type=list, nargs='+', default=None)
    parser.add_argument('--channel', type=int, help='channel num of imgs', default=None)
    parser.add_argument('--img_size', type=int, help='height/width of imgs', default=None)
    

    parser.add_argument('--base_dir', type=str, help='dir of datasets', default=None)
    parser.add_argument('--output_dir', type=str, help='dir of datasets', default=None)  # ./data/temp


    return parser.parse_args(argv)



if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    
    if args.shuffle:
        shuffle()
    elif args.unshuffle:
        unshuffle()
