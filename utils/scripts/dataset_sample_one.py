import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image

sys.path.append(os.getcwd())
from utils.datasets import get_dataset_pkl

def main():
    idxes = range(args.sample_num)  # gtsrb:[0, 7, 9, 11, 13, 14, 16, 22]    gtsrb: [48]  cifar10: [4998]
    fpath = os.path.join(args.output_dir, args.dataset)  # + '_' + args.subset
    os.makedirs(fpath, exist_ok=True)
    

    name = '_'.join(args.subset.split('_')[:-1]) if args.name is None else args.name
    if os.path.exists(os.path.join(fpath, name + '_0.png')):
        return

    ds = get_dataset_pkl(os.path.join(args.dataset, args.subset), args.split)

    for i in idxes:
        img, tgt = ds.data[i]
        # img = Image.fromarray((img*255).astype(np.uint8).squeeze())
        # 
        if len(idxes) > 1:
            save_dir = os.path.join(fpath, name + '_' + str(i) + '.png')
        else:
            save_dir = os.path.join(fpath, name+'.png')
        img.save(save_dir)
    # print('img saved to ', fpath)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='name of the dataset', default='cifar10')
    parser.add_argument('--subset', type=str, help='subset of dataset', default='clean')
    parser.add_argument('--split', type=str, help='train/test/val/dev', default='train')
    parser.add_argument('--sample_num', type=int, help='sample num', default=10)

    # parser.add_argument('--base_dir', type=str, help='dir of datasets', default='temp')
    parser.add_argument('--name', type=str, help='img file name', default=None)  # ./data/temp
    parser.add_argument('--output_dir', type=str, help='dir of datasets', default='data/temp')  # ./data/temp

    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
