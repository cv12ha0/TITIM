import argparse
import copy
import warnings
import os
import sys
import json
import time
import pickle
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from utils.models import get_model
from utils.datasets import get_dataset, get_img_shape, get_num_classes
from utils.misc import set_random_seed, get_device, get_nw, get_optimizer, get_scheduler, ImageProcess, AugmentationPost
from utils.backdoor import WaNet


def train(args):
    warnings.filterwarnings("ignore", category=UserWarning)
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    set_random_seed(args.seed)
    device = get_device(args.gpu)
    args.img_shape = get_img_shape(args.dataset, args.img_shape, args.channel, args.img_size)
    args.num_classes = get_num_classes(args.dataset, args.num_classes)
    args.num_workers = get_nw(args.num_workers)
    # load trigger
    trigger = WaNet([args.target], args.img_shape, args.cross_ratio, args.s, args.k, args.grid_rescale, args.fixed)
    trigger_config = trigger.config

    if args.output_dir is None:
        args.output_dir = '_'.join([args.prefix, args.model, args.dataset, trigger.name, str(args.inject_ratio), 'e'+str(args.epochs), args.suffix]).strip('_')  
    # overwrite all / overwrite unfinished / skip
    if os.path.exists(os.path.join('res', args.output_dir)):
        if (args.mode in ['overwrite']) or (args.mode in ['cover'] and not os.path.exists(os.path.join('res', args.output_dir, 'stats.json'))):
            import shutil
            print('removing existing dir: ', args.output_dir)
            shutil.rmtree(os.path.join('res', args.output_dir))
    os.makedirs(os.path.join('res', args.output_dir), exist_ok=False)
        
    print("device:", args.gpu, "  num_workers:", args.num_workers, '  lr:', args.lr, '  optimizer:', args.optimizer, '  batch_size:', args.batch_size)
    print("dataset:", args.dataset + '/' + args.subset)
    print("trigger config: ", trigger_config)
    print('output dir: ', args.output_dir)
    
    # model, freeze params if fine-tune
    model = get_model(args.model, args.img_shape, args.num_classes, device, )
    if args.pretrain:
        model.load_state_dict(torch.load('data/model/' + args.model + '.pth', map_location=device), strict=False)
    # model = torch.load('./res/googlenet_lr0.01_e40_cifar10_clean_rhf_cosann_sgd/model.pth', map_location='cuda:0')  # ./res/resnet18_lr1e-05_e20_mnist_clean_/model.pth
    print(model.config)

    criterion = torch.nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = get_optimizer(args.optimizer, params, args.lr)
    scheduler = get_scheduler(args.scheduler, optimizer, args.epochs, args.lr_step, args.lr_factor, args.lr_patience, args.lr_min)

    ds_train = get_dataset(os.path.join(args.dataset, 'clean'), 'train', fmt=args.ds_fmt)
    ds_vals = [get_dataset(os.path.join(args.dataset, 'clean'), args.split_val, fmt=args.ds_fmt),  # test val  
               get_dataset(os.path.join(args.dataset, args.subset), args.split_val2, fmt=args.ds_fmt), ]  # test val args.split_val

    # preprocess
    image_process = ImageProcess(args.img_shape)
    augmentation = AugmentationPost(args.img_shape)

    ds_train = ds_train.map(image_process)
    ds_vals = [ds.map(image_process) for ds in ds_vals]
    dl_train = DataLoader(ds_train, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False, pin_memory=True)
    dl_vals = [DataLoader(ds_val, args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True) for ds_val in ds_vals]

    # train & eval
    model, log = fit_wanet(trigger, args.inject_ratio, model, args.epochs, criterion, optimizer, dl_train, dl_vals, scheduler, device, args.batch_size, output_dir=args.output_dir, return_log=True, plot=True, augmentation=augmentation, disable_prog=args.disable_prog)
    loss_acc, acc, time_acc = evaluate_wanet(trigger, 0, model, dl_vals[0], criterion, device, batch_size=args.batch_size, verbose="final acc", output_info=True)
    loss_asr, asr, time_asr = evaluate_wanet(trigger, 1, model, dl_vals[1], criterion, device, batch_size=args.batch_size, verbose="final asr", output_info=True)

    # save
    ds_info = json.load(open(os.path.join('data', args.dataset, args.subset, 'stats.json'), 'r'))
    ratio = ds_info['train']['inject_ratio']
    torch.save(model, os.path.join('res', args.output_dir, 'model.pth'))
    with open(os.path.join('res', args.output_dir, 'stats.json'), 'a+') as f:
        json.dump(dict(dataset=args.dataset, subset=args.subset, optimizer=args.optimizer, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
                       inject_ratio=ratio, acc=acc, asr=asr, loss_acc=loss_acc, loss_asr=loss_asr,
                       model_config=model.config, trigger_config=trigger_config,
                       training_stats=log.get_dict()), f, indent=4)
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

    # wanet config
    parser.add_argument('--target', type=int, help='attack target', default=None)
    parser.add_argument('--inject_ratio', type=float, help='inject ratio', default=0)
    parser.add_argument('--cross_ratio', type=float, help='cross ratio of wanet', default=0)
    parser.add_argument('--s', type=float, help='s of wanet', default=0.5)
    parser.add_argument('--k', type=int, help='k of wanet', default=4)
    parser.add_argument('--grid_rescale', type=float, help='grid rescale of wanet', default=1.)
    parser.add_argument('--fixed', action='store_true', help='use a fixed pattern')

    # training config
    parser.add_argument('--model', type=str, help='model name', default='resnet18')
    parser.add_argument('--pretrain', action='store_true', help="use pretrained model if set true")
    parser.add_argument('--dataset', type=str, help='mnist/cifar10/gtsrb/tiny_imagenet', default='cifar10')
    parser.add_argument('--subset', type=str, help='name of the dataset', default='clean')

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



def fit_wanet(trigger, inject_ratio, model, epochs, criterion, optimizer, dl_train, dl_vals, scheduler, device, batch_size=32,
              output_dir=None, save_last=True, return_log=False, plot=False, augmentation=None, disable_prog=False):
    from utils.misc import Log
    print("\ntraining starts at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    log = Log()
    os.makedirs(os.path.join('res', output_dir, 'weights'), exist_ok=True)
    for epoch in range(epochs):
        # train
        model.train()
        running_loss, counter = 0, 0
        prog_bar = tqdm(dl_train, file=sys.stdout, leave=False, disable=disable_prog)
        time_start = time.perf_counter()
        for step, (batch_x, batch_y) in enumerate(prog_bar):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            baych_x, batch_y = trigger.apply_batch(batch_x, batch_y, inject_ratio, trigger.cross_ratio * inject_ratio, device=device)
            baych_x, batch_y = augmentation.apply_batch(batch_x, batch_y)

            optimizer.zero_grad()
            pre = model(batch_x)
            loss = criterion(pre, batch_y)

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            running_loss += loss.item()
            counter += 1
            prog_bar.desc = "train epoch[{}/{}] loss:{:.4f} lr:{:.6f}" \
                .format(epoch+1, epochs, loss, optimizer.state_dict()['param_groups'][0]['lr'])
        scheduler.step()

        # val
        model.eval()
        loss_acc, acc, time_acc = evaluate_wanet(trigger, 0, model, dl_vals[0], criterion, device, batch_size, "acc", disable_prog)
        loss_asr, asr, time_asr = evaluate_wanet(trigger, 1, model, dl_vals[1], criterion, device, batch_size, "asr", disable_prog)

        # logging
        time_train = time.perf_counter() - time_start  # prog_bar.format_dict['elapsed']
        log.loss_train.append(running_loss / counter)
        log.time_train.append(time_train)
        log.acc.append(acc)
        log.loss_acc.append(loss_acc)
        log.time_acc.append(time_acc)
        log.asr.append(asr)
        log.loss_asr.append(loss_asr)
        log.time_asr.append(time_asr)
        stat_cur = "{}: epoch[{:>3}/{}] loss:{:.4f} lr:{:.6f} elapsed:{:.2f}   loss:{:.6f} acc:{:.4f} elapsed:{:.2f}   loss:{:.6f} asr:{:.4f} elapsed:{:.2f}".format(
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), epoch + 1, epochs, running_loss / counter, optimizer.state_dict()['param_groups'][0]['lr'], time_train, loss_acc, acc, time_acc, loss_asr, asr, time_asr)
        log.stats.append(stat_cur)
        log.dump_csv(output_dir, epoch, [running_loss/counter, loss_acc, loss_asr, acc, asr, time_train, time_acc, time_asr])
        print(stat_cur)

        if plot:
            log.plot(output_dir, plot_all=True)
        
        if output_dir is not None and log.acc[-1] > log.best_acc:
            log.best_acc = log.acc[-1]
            ckpt = {'epoch': epoch, 'acc': acc, 'asr': asr, 'model': deepcopy(model), 'optimizer': optimizer.state_dict(), 'date': datetime.now().isoformat()}
            torch.save(ckpt, os.path.join('res', output_dir, 'weights', 'best.pth'))
        # print()
    if output_dir is not None and save_last:
        ckpt = {'epoch': epoch, 'acc': acc, 'asr': asr, 'model': deepcopy(model), 'optimizer': optimizer.state_dict(), 'date': datetime.now().isoformat()}
        torch.save(ckpt, os.path.join('res', output_dir, 'weights', 'last.pth'))
    print("training finished at " + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    if return_log:
        return model, log
    else:
        return model


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
    # import ssl
    # ssl._create_default_https_context = ssl._create_unverified_context
    args = parse_arguments(sys.argv[1:])
    args.split_val2 = args.split_val if args.split_val2 is None else args.split_val2
    
    train(args)
    # print('fin')



