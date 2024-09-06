import argparse
import warnings
import os
import sys
import json
import pickle
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils.models import get_model
from utils.datasets import get_dataset, get_img_shape, get_num_classes
from utils.misc import set_random_seed, get_device, get_nw, get_optimizer, get_scheduler, fit, evaluate, ImageProcess, Augmentation
from utils.backdoor import FakeTrigger


def train(args):
    warnings.filterwarnings("ignore", category=UserWarning)
    set_random_seed(args.seed)
    device = get_device(args.gpu)
    args.img_shape = get_img_shape(args.dataset, args.img_shape, args.channel, args.img_size)
    args.num_classes = get_num_classes(args.dataset, args.num_classes)
    args.num_workers = get_nw(args.num_workers)
    if args.output_dir is None:
        args.output_dir = '_'.join([args.prefix, args.model, args.dataset, args.subset, 'e'+str(args.epochs), args.suffix]).strip('_')  
    # overwrite all / overwrite failed / skip
    if os.path.exists(os.path.join('res', args.output_dir)):
        if (args.mode in ['overwrite']) or (args.mode in ['cover'] and not os.path.exists(os.path.join('res', args.output_dir, 'stats.json'))):
            import shutil
            print('removing existing dir: ', args.output_dir)
            shutil.rmtree(os.path.join('res', args.output_dir))
    os.makedirs(os.path.join('res', args.output_dir), exist_ok=False)
    # load trigger
    if os.path.exists(os.path.join('data', args.dataset, args.subset, 'trigger_config.json')):
        trigger_config = json.load(open(os.path.join('data', args.dataset, args.subset, 'trigger_config.json'), 'r'))
    else:
        try:
            trigger = pickle.load(open(os.path.join('data', args.dataset, args.subset, 'trigger.pkl'), 'rb'))
        except FileNotFoundError:
            trigger = FakeTrigger(0)
        trigger_config = trigger.config
    print("device:", args.gpu, "  num_workers:", args.num_workers, '  lr:', args.lr, '  optimizer:', args.optimizer, '  batch_size:', args.batch_size)
    print("dataset:", args.dataset + '/' + args.subset)
    print("trigger config: ", trigger_config)
    print('output dir: ', args.output_dir)
    
    # model, freeze params if fine-tune
    model = get_model(args.model, args.img_shape, args.num_classes, device, )
    if args.pretrain:
        model.load_state_dict(torch.load('data/model/' + args.model + '.pth', map_location=device), strict=False)
    print(model.config)

    criterion = torch.nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = get_optimizer(args.optimizer, params, args.lr)
    scheduler = get_scheduler(args.scheduler, optimizer, args.epochs, args.lr_step, args.lr_factor, args.lr_patience, args.lr_min)

    ds_train = get_dataset(os.path.join(args.dataset, args.subset), 'train', fmt=args.ds_fmt)
    ds_vals = [get_dataset(os.path.join(args.dataset, 'clean'), args.split_val, fmt=args.ds_fmt),  # test val
               get_dataset(os.path.join(args.dataset, args.subset_val), args.split_val2, fmt=args.ds_fmt), ]  # test val args.split_val

    # preprocess
    image_process = ImageProcess(args.img_shape)
    augmentation = Augmentation(args.img_shape)
    
    ds_train = ds_train.map(augmentation).map(image_process)
    ds_vals = [ds.map(image_process) for ds in ds_vals]
    dl_train = DataLoader(ds_train, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False, pin_memory=True)  # , persistent_workers=True
    dl_vals = [DataLoader(ds_val, args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True) for ds_val in ds_vals]

    # train & eval
    model, log = fit(model, args.epochs, criterion, optimizer, dl_train, dl_vals, scheduler, device, args.batch_size, output_dir=args.output_dir, return_log=True, plot=True, disable_prog=args.disable_prog)
    loss_acc, acc, time_acc = evaluate(model, dl_vals[0], criterion, device, batch_size=args.batch_size, verbose="final acc", output_info=True)
    loss_asr, asr, time_asr = evaluate(model, dl_vals[1], criterion, device, batch_size=args.batch_size, verbose="final asr", output_info=True)

    # save
    ds_info = json.load(open(os.path.join('data', args.dataset, args.subset, 'stats.json'), 'r'))
    ratio = ds_info['train']['inject_ratio']
    torch.save(model, os.path.join('res', args.output_dir, 'model.pth'))
    with open(os.path.join('res', args.output_dir, 'stats.json'), 'a+') as f:
        json.dump(dict(dataset=args.dataset, subset=args.subset, optimizer=args.optimizer, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
                       inject_ratio=ratio, acc=acc, asr=asr, loss_acc=loss_acc, loss_asr=loss_asr,
                       model_config=model.config, trigger_config=trigger_config,
                       training_stats=log.get_dict()), f, indent=4)
    os.makedirs('logs', exist_ok=True)
    with open(args.log, 'a') as f:
        f.write('\n' + args.output_dir + '\t' + str(round(asr*100, 2)) + '\t' + str(round(acc*100, 2)))
    print('file saved to res/' + args.output_dir + '/')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, help='GPU id', default='0')
    parser.add_argument('--seed', type=int, help='global seed', default=0)
    parser.add_argument('--mode', type=str, help='overwrite/cover/normal', default='normal')
    parser.add_argument('--ds_fmt', type=str, help='dataset format: pickle(pkl)/folder/offic', default='pkl')
    parser.add_argument('--split_val', type=str, help='train/test/val/dev', default='test')
    parser.add_argument('--split_val2', type=str, help='train/test/val/dev', default=None)
    # parser.add_argument('--split', type=str, help='train/test/val/dev', default=None)
    # parser.add_argument('--plot', action='store_true', help='plot training loss curve')
    parser.add_argument('--disable_prog', action='store_true', help='disable progress bar')
    parser.add_argument('--config', type=str, help='dir of config file', default='imdb_test')
    parser.add_argument('--log', type=str, help='path to log file', default='logs/asrlog.tsv')

    # training config
    parser.add_argument('--model', type=str, help='model name', default='resnet18')
    parser.add_argument('--pretrain', action='store_true', help="use pretrained model if set true")
    parser.add_argument('--dataset', type=str, help='mnist/cifar10/gtsrb/tiny_imagenet', default='cifar10')
    parser.add_argument('--subset', type=str, help='name of the dataset', default='clean')
    parser.add_argument('--subset_val', type=str, help='subset for validation', default=None)

    parser.add_argument('--num_classes', type=int, help='classes num of the dataset', default=None)
    parser.add_argument('--img_shape', type=list, nargs='+', default=None)
    parser.add_argument('--channel', type=int, help='channel num of imgs', default=None)
    parser.add_argument('--img_size', type=int, help='height/width of imgs', default=None)
    parser.add_argument('--num_workers', '--nw', type=int, help='num of workers', default=4)
    parser.add_argument('--batch_size', '--bs', type=int, help='batch size', default=128)
    parser.add_argument('--epochs', type=int, help='num of epochs', default=40)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-3)

    parser.add_argument('--optimizer', type=str, help='adam/sgd/adamdelta', default='sgd')
    parser.add_argument('--scheduler', type=str, help='step/plateau/cosann', default='cosann')
    parser.add_argument('--lr_step', type=int, help='for step', default=10)
    parser.add_argument('--lr_factor', type=float, help='for step/plateau', default=0.1)
    parser.add_argument('--lr_patience', type=int, help='for plateau', default=5)
    parser.add_argument('--lr_min', type=float, help='for plateau', default=0.5e-6)


    parser.add_argument('--base_dir', type=str, help='dir of datasets', default='./res')
    parser.add_argument('--output_dir', type=str, help='dir of datasets', default=None)  # ./data/temp
    parser.add_argument('--prefix', type=str, help='prefix of output dir', default='')
    parser.add_argument('--suffix', type=str, help='suffix of output dir', default='')

    return parser.parse_args(argv)


if __name__ == '__main__':
    # import ssl
    # ssl._create_default_https_context = ssl._create_unverified_context
    args = parse_arguments(sys.argv[1:])
    args.split_val2 = args.split_val if args.split_val2 is None else args.split_val2
    args.subset_val = args.subset if args.subset_val is None else args.subset_val
    
    train(args)
    # print('fin')

