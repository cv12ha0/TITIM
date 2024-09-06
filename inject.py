import argparse
import warnings
import os
import sys
import json
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch

from utils.datasets import get_dataset, get_img_shape, get_num_classes, dump_ds_pd
from utils.models import get_model
from utils.misc import set_random_seed, get_device
from utils.backdoor import dump_trigger, dump_trigger_config, FakeTrigger, Patch, BadNets, Blended, Styled, WaNet, Compress, SIG


def inject(args):
    warnings.filterwarnings("ignore", category=UserWarning)
    set_random_seed(args.seed)
    args.img_shape = get_img_shape(args.dataset, args.img_shape, args.channel, args.img_size)
    args.num_classes = get_num_classes(args.dataset, args.num_classes)
    trigger = sel_trigger(args)
    # set output dir
    if args.output_dir is None:
        subset = '_'.join([args.prefix, trigger.name, str(args.ratio), args.suffix]).strip('_')
        args.output_dir = os.path.join('data', args.dataset, subset, )
    os.makedirs(args.output_dir, exist_ok=True)
    # load/save trigger
    if os.path.exists(os.path.join(args.output_dir, 'trigger.pkl')):
        trigger = pickle.load(open(os.path.join(args.output_dir, 'trigger.pkl'), 'rb'))  # load trigger if exists
        print('using existing trigger.')
    elif args.dump_trigger:
        dump_trigger(trigger, args.output_dir)
    dump_trigger_config(trigger, args.output_dir)
    
    print("trigger config: ", trigger.config)
    print("output_dir: ", args.output_dir)
    print("dataset: ", args.dataset)
    print("num_classes: ", args.num_classes)
    print("img_shape: ", args.img_shape)


    def inject_split(trigger_func, split, ratio, ratio_start=0.0, ds=None, info=None, dump=True):
        print('injecting', args.dataset, split, ' ratio='+str(ratio), ' ratio_start='+str(ratio_start))
        if ds is None:
            ds = get_dataset(os.path.join(args.dataset, 'clean'), split, fmt=args.ds_fmt)  # clean badnets_mixtest
        ds_dump = []
        start_idx = int(len(ds) * ratio_start)
        poison_idx = random.sample(range(len(ds)), int(len(ds) * ratio)) if args.shuffle else list(range(len(ds))[start_idx:int(len(ds) * ratio) + start_idx])
        prog_bar = tqdm(ds, file=sys.stdout)
        for idx, item in enumerate(prog_bar, 0):
            if idx in poison_idx:
                item = trigger_func(item)
                ds_dump.insert(0, item)
            else:
                ds_dump.append(item)
        if info is None:
            info = {split: {'dataset': args.dataset, 'img_shape': args.img_shape, 'inject_ratio': ratio, 'trigger_config': trigger.config, }}
        if dump:
            dump_ds_pd(ds_dump, args.output_dir, split, info, columns=['image', 'label'])
        return ds_dump, info


    def inject_split_folder(trigger_func, split, ratio, ratio_start=0.0, ds=None, info=None, dump=True):
        print('injecting', args.dataset, split, ' ratio='+str(ratio), ' ratio_start='+str(ratio_start))
        if ds is None:
            ds = get_dataset(os.path.join(args.dataset, 'clean'), split, fmt=args.ds_fmt)  # clean
        if os.path.exist(os.path.join(args.output_dir, split, 'labels.csv')):
            label_info = pd.read_csv(os.path.join(args.output_dir, split, 'labels.csv'), sep='\t').values.tolist()
        else:
            label_info = [["%06d" % i, -1] for i in range(len(ds))]
        start_idx = int(len(ds) * ratio_start)
        poison_idx = list(range(len(ds))[start_idx:int(len(ds) * ratio) + start_idx])
        prog_bar = tqdm(ds, file=sys.stdout)
        for idx, item in enumerate(prog_bar, 0):
            if idx in poison_idx:
                item = trigger_func(item)
            img_cur, label_cur = item
            img_cur = Image.fromarray((img_cur * 255).astype(np.uint8))
            img_cur.save(os.path.join(args.output_dir, split, "%06d.png" % idx))
            label_info.append(["%06d", label_cur])
            label_info[idx] = label_cur
        if info is None:
            info = {split: {'dataset': args.dataset, 'img_shape': args.img_shape, 'inject_ratio': ratio, 'trigger_config': trigger.config, }}
        # dump info
        with open(os.path.join(args.output_dir, split, 'labels.csv'), 'a+') as f:
            f.write(label_info)
        if dump:
            try:
                info_orig = json.load(open(os.path.join(args.output_dir, 'stats.json'), 'r'))
            except FileNotFoundError:
                info_orig = {}
            info_orig.update(info)
            json.dump(info_orig, open(os.path.join(args.output_dir, 'stats.json'), 'w'), indent=4)
            info = info_orig

        return None, info

    if args.trigger.lower() in ['wanet']:
        ds, info = inject_split(trigger.apply_a, args.split, args.ratio, args.ratio_start, dump=False)
        ds, info = inject_split(trigger.apply_n, args.split, args.cross_ratio*args.ratio, args.ratio+args.ratio_start, ds=ds, info=info)
        if args.split_val is not None:
            inject_split(trigger.apply_a, args.split_val, 1, 0)
            inject_split(trigger.apply_n, args.split_val+'_n', 1, 0)
    elif args.trigger.lower() in ['compose']:
        ds, info = inject_split(trigger.apply1, args.split, args.ratio, args.ratio_start, dump=False)
        ds, info = inject_split(trigger.apply2, args.split, args.ratio, args.ratio+args.ratio_start, ds=ds, info=info)
        if args.split_val is not None:
            inject_split(trigger.apply0, args.split_val, 1, 0)
            inject_split(trigger.apply1, args.split_val+'_1', 1, 0)
            inject_split(trigger.apply2, args.split_val+'_2', 1, 0)

    else:
        inject_split(trigger.__call__, args.split, args.ratio, args.ratio_start)
        if args.split_val is not None:
            inject_split(trigger.__call__, args.split_val, 1, 0)





def sel_trigger(args):
    name = args.trigger.lower()
    if name in ['fake', 'clean']:
        trigger = FakeTrigger(1)
    elif name in ['patch']:
        trigger = Patch([args.target], args.img_shape, args.pattern, args.mask_ratio, args.patch_size, args.patch_loc)
    elif name in ['badnets']:
        trigger = BadNets([args.target], args.img_shape, args.block_size, args.block_num, args.mask_ratio, args.pattern_per_target, args.fixed)
    elif name in ['blended', 'blend']:
        trigger = Blended([args.target], args.img_shape, args.pattern, args.mask_ratio)  # noise hellokitty
    elif name in ['styled', 'filter']:
        trigger = Styled([args.target], args.img_shape, args.filter, args.framed)  # gotham  nashville  kelvin  toatser  lomo
    elif name in ['wanet']:
        trigger = WaNet([args.target], args.img_shape, args.cross_ratio, args.s, args.k, args.grid_rescale, args.fixed)
    elif name in ['sig']:
        trigger = SIG([args.target], args.img_shape, args.delta, args.f)
    elif name in ['compress']:
        trigger = Compress([args.target], args.img_shape, args.compress_alg, args.compress_quality)
    return trigger


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, help='GPU id', default='0')
    parser.add_argument('--seed', type=int, help='global seed', default=0)
    parser.add_argument('--shuffle', action='store_true', help='gen random pos idx if true')
    parser.add_argument('--config', type=str, help='dir of config file', default='imdb_test')
    parser.add_argument('--ds_fmt', type=str, help='dataset format: pickle(pkl)/folder/offic', default='pkl')
    parser.add_argument('--split', type=str, help='train/test/val/dev', default='train')
    parser.add_argument('--split_val', type=str, help='train/test/val/dev', default='test')

    # dataset & inject config
    parser.add_argument('--dataset', type=str, help='name of the dataset', default='cifar10')
    parser.add_argument('--num_classes', type=int, help='classes num of the dataset', default=None)
    parser.add_argument('--img_shape', type=list, nargs='+', default=None)
    parser.add_argument('--channel', type=int, help='channel num of imgs', default=None)
    parser.add_argument('--img_size', type=int, help='height/width of imgs', default=None)
    parser.add_argument('--ratio', type=float, help='inject ratio', default=0)
    parser.add_argument('--ratio_start', type=float, help='inject index from ... ', default=0)
    parser.add_argument('--target', type=int, help='attack target', default=None)
    
    # backdoor config
    parser.add_argument('--trigger', type=str, help='trigger type', default='clean')
    parser.add_argument('--fixed', action='store_true', help='use a fixed pattern')
    parser.add_argument('--dump_trigger', action='store_true', help='dump trigger obj to output folder')
    # parser.add_argument('--use_existing_trigger', action='store_true', help='use existing trigger in target folder')
    parser.add_argument('--mask_ratio', '--mr', type=float, help='trigger opacity', default=1)
    parser.add_argument('--patch_size', type=int, help='trigger size of patch', default=4)
    parser.add_argument('--patch_loc', type=int, help='trigger location of patch', default=28)
    parser.add_argument('--block_size', type=int, help='trigger size of square & badnets', default=4)
    parser.add_argument('--margin', type=int, help='loc of block', default=1)
    parser.add_argument('--color', type=str, help='color or hex', default='white')
    parser.add_argument('--block_num', type=int, help='patch num of badnets', default=3)
    parser.add_argument('--pattern_per_target', '--ppt', type=int, help='pattern per target', default=1)
    parser.add_argument('--pattern', type=str, help='pattern of blended', default='hellokitty')
    parser.add_argument('--filter', type=str, help='filter name of styled', default='gotham')
    parser.add_argument('--framed', type=bool, help='whether to use filter frame image', default=False)
    parser.add_argument('--cross_ratio', type=float, help='cross ratio of wanet', default=0)
    parser.add_argument('--s', type=float, help='s of wanet', default=0.5)
    parser.add_argument('--k', type=int, help='k of wanet', default=4)
    parser.add_argument('--grid_rescale', type=float, help='grid rescale of wanet', default=1.)
    parser.add_argument('--model', type=str, help='model arch for clean label', default='resnet18')
    parser.add_argument('--model_dir', type=str, help='model dir for clean label', default=None)
    parser.add_argument('--perturb', type=str, help='perturbation of clean label', default='none')
    parser.add_argument('--cl_trigger', type=str, help='trigger of clean label', default='badnets')
    parser.add_argument('--issba_epochs', type=int, help='training epochs of issba encoder', default=20)
    parser.add_argument('--issba_epochs_secret', type=int, help='training epochs of issba encoder', default=5)
    parser.add_argument('--secret', type=str, help='secret string of issba', default='#secret')
    parser.add_argument('--secret_bits', type=str, help='secret bits of issba', default='00110101')
    parser.add_argument('--use_bch', type=bool, help='use secret str instead of bits', default=False)
    parser.add_argument('--res_ratio', type=float, help='redisual mixing ratio of issba', default=1.0)
    parser.add_argument('--issba_pretrain', type=bool, help='whether to use pretrained encoder in issba', default=True)
    parser.add_argument('--issba_fname', type=str, help='pretrained encoder file name, auto-gen when set to None', default=None)
    parser.add_argument('--compress_alg', type=str, help='compress algorithm', default='none')
    parser.add_argument('--compress_quality', type=int, help='compress quality', default=100)
    parser.add_argument('--delta', type=int, help='amplitude of backdoor signal in sig', default=40)
    parser.add_argument('--f', type=int, help='frequency of backdoor signal in sig', default=6)


    parser.add_argument('--base_dir', type=str, help='dir of datasets', default='./data')
    parser.add_argument('--output_dir', type=str, help='dir of datasets', default=None)  # ./data/temp
    parser.add_argument('--prefix', type=str, help='prefix of output dir', default='')
    parser.add_argument('--suffix', type=str, help='suffix of output dir', default='')


    return parser.parse_args(argv)


if __name__ == '__main__':
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    args = parse_arguments(sys.argv[1:])
    inject(args)
    # print('fin')
    print()
