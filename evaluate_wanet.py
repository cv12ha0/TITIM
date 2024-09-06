import argparse
import copy
import warnings
import os
import sys
import pickle
import time
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from utils.models import get_model
from utils.datasets import get_dataset, get_img_shape, get_num_classes
from utils.misc import set_random_seed, get_device, get_nw, ImageProcess, evaluate
from utils.backdoor import FakeTrigger, Square, BadNets, WaNet


def main(args):
    warnings.filterwarnings("ignore", category=UserWarning)
    set_random_seed(args.seed)
    device = get_device(args.gpu)
    args.img_shape = get_img_shape(args.dataset, args.img_shape, args.channel, args.img_size)
    args.num_classes = get_num_classes(args.dataset, args.num_classes)
    args.num_workers = get_nw(args.num_workers)
    if args.name is None:
        args.name = '_'.join([args.prefix, args.model_dir, 't'+args.subset, args.suffix]).strip('_')
    criterion = torch.nn.CrossEntropyLoss()

    model_dir = os.path.join('res', args.model_dir, 'weights', args.model_name+'.pth')
    model = torch.load(model_dir, map_location=device)['model']  # .to(device)
    ds = get_dataset(os.path.join(args.dataset, 'clean'), args.split, fmt=args.ds_fmt)  # args.subset
    trigger = WaNet([args.target], args.img_shape, args.cross_ratio, args.s, args.k, args.grid_rescale, args.fixed)
    print("model: ", model_dir)
    print("dataset: ", os.path.join(args.dataset, args.subset, args.split))
    print("trigger config: ", trigger.config)

    # preprocess
    image_process = ImageProcess(args.img_shape)
    ds = ds.map(image_process)
    dl = DataLoader(ds, args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)
    # eval
    loss_acc, acc, time_acc = evaluate_wanet(trigger, 1, model, dl, criterion, device, args.batch_size, "acc", False, True)

    with open('logs/' + args.log, 'a') as f:
        f.write('\n' + args.name + '\t' + str(round(acc*100, 2)))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, help='GPU id', default='0')
    parser.add_argument('--seed', type=int, help='global seed', default=0)    
    parser.add_argument('--config', type=str, help='dir of config file', default='imdb_test')
    parser.add_argument('--ds_fmt', type=str, help='dataset format: pickle(pkl)/folder/offic', default='pkl')
    parser.add_argument('--log', type=str, help='path to log file', default='asrlog.tsv')  # logs/asrlog.tsv

    # wanet config
    parser.add_argument('--target', type=int, help='attack target', default=None)
    parser.add_argument('--inject_ratio', type=float, help='inject ratio', default=0)
    parser.add_argument('--cross_ratio', type=float, help='cross ratio of wanet', default=0)
    parser.add_argument('--s', type=float, help='s of wanet', default=0.5)
    parser.add_argument('--k', type=int, help='k of wanet', default=4)
    parser.add_argument('--grid_rescale', type=float, help='grid rescale of wanet', default=1.)
    parser.add_argument('--fixed', action='store_true', help='use a fixed pattern')

    # config
    parser.add_argument('--model_dir', type=str, default='./res')
    parser.add_argument('--model_name', type=str, default='best')  # last/best
    parser.add_argument('--dataset', type=str, help='mnist/cifar10/gtsrb/tiny_imagenet', default='cifar10')
    parser.add_argument('--subset', type=str, help='name of the dataset', default='clean')
    parser.add_argument('--split', type=str, help='train/test/val/dev', default=None)

    parser.add_argument('--num_classes', type=int, help='classes num of the dataset', default=None)
    parser.add_argument('--img_shape', type=list, nargs='+', default=None)
    parser.add_argument('--channel', type=int, help='channel num of imgs', default=None)
    parser.add_argument('--img_size', type=int, help='height/width of imgs', default=None)
    parser.add_argument('--num_workers', '--nw', type=int, help='num of workers', default=4)
    parser.add_argument('--batch_size', '--bs', type=int, help='batch size', default=128)

    parser.add_argument('--name', type=str, help='output_name', default=None)  # ./data/temp
    parser.add_argument('--prefix', type=str, help='prefix of output name', default='')
    parser.add_argument('--suffix', type=str, help='suffix  of output name', default='')
    return parser.parse_args(argv)



def evaluate_wanet(trigger, inject_ratio, model, dl_val, criterion, device, batch_size=32, verbose="acc", disable=False, output_info=False, transform=None):
    model.eval()
    num_total, num_acc = 0, 0
    prog_bar = tqdm(dl_val, file=sys.stdout, leave=False, disable=disable)
    time_start = time.perf_counter()
    with torch.no_grad():
        for _, (batch_x, batch_y) in enumerate(prog_bar, 0):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_x, batch_y = trigger.apply_batch(batch_x, batch_y, inject_ratio, 0, device=device)
            pre = model(batch_x)
            loss = criterion(pre, batch_y).item()
            pre = torch.max(pre, dim=1)[1]
            # pre = torch.round(pre)  # BCELoss
            num_acc += torch.eq(pre, batch_y).sum().item()
            num_total += batch_y.size(0)
            prog_bar.desc = "valid[{}] ".format(verbose)
    acc = num_acc / num_total
    elapsed = time.perf_counter() - time_start  # prog_bar.format_dict['elapsed']
    if output_info:
        print('loss: %.6f  %s: %.4f (%d/%d)' % (loss, verbose, acc, num_acc, num_total))
    return loss, acc, elapsed




if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    # torch.multiprocessing.set_start_method('spawn')
    main(args)
