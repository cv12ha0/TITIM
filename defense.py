import argparse
import copy
import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch

from utils.models import get_model
from utils.datasets import get_dataset, get_img_shape, get_num_classes, TorchCifar10, merge_ds
from utils.backdoor import NeuralCleanse, KArm, ActivationClustering, SpectralSignature, STRIP, ABL, ScaleUp, FeatureRE
from utils.misc import set_random_seed, get_device, get_nw, get_optimizer, get_regularization, ImageProcess, evaluate


def main(args):
    set_random_seed(args.seed)
    device = get_device(args.gpu)
    args.img_shape = get_img_shape(args.dataset, args.img_shape, args.channel, args.img_size)
    args.num_classes = get_num_classes(args.dataset, args.num_classes)
    args.num_workers = get_nw(args.num_workers)
    print("device:", args.gpu, "  img_shape:", args.img_shape, "  num_classes:", args.num_classes)

    ds = get_dataset(os.path.join(args.dataset, args.subset), args.split, fmt=args.ds_fmt)  # test_sample_20
    ds_vals = [get_dataset(os.path.join(args.dataset, 'clean'), args.split_val, fmt=args.ds_fmt),  # test val
               get_dataset(os.path.join(args.dataset, args.subset_val), args.split_val2, fmt=args.ds_fmt), ]  # test val args.split_val
    image_process = ImageProcess(args.img_shape)
    ds = ds.map(image_process)
    ds_vals = [ds.map(image_process) for ds in ds_vals]
    # dl = torch.utils.data.DataLoader(ds, args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)
    print("dataset: ", args.dataset, '    subset:', args.subset)

    model_dir = None
    if args.model_dir is not None:
        model_dir = os.path.join('res', args.model_dir, 'weights', args.model_name+'.pth')
        model = torch.load(model_dir, map_location=device)['model']
        print('using model from:', model_dir)
    else:
        model = get_model(args.model, args.img_shape, args.num_classes, device)
        print('using clean model:', args.model)

    defense(args, ds, ds_vals, model, model_dir, device)
    print()



def defense(args, ds, ds_vals, model, model_dir, device):
    name = args.defense.lower()
    tgt_ls = [args.target] if args.target is not None else None
    result = None

    if name in ['ac', 'activation_clustering']:
        # Activation Clustering
        ac = ActivationClustering(args.img_shape, args.num_classes, decomposition=args.decomposition, device=device)
        print(ac.config)
        res = ac.clustering(ds, model, layer_name=args.layer_name)  # layer4  conv_block3  avgpool
        ac.dump_stats(os.path.join('data', args.dataset, args.subset, 'ActivationClustering'), model.config, model_dir)
        # log
        name_cur = '_'.join([args.dataset, args.subset])
        result = [name_cur, args.layer_name] + [round(i, 6) for i in res['silhouette']] + [round(i, 6) for i in res['size']]

    elif name in ['ss', 'spectral_signature']:
        # Spectral Signature
        ss = SpectralSignature(args.img_shape, args.num_classes, tgt_ls=tgt_ls, device=device)
        print(ss.config)
        res = ss.filter(ds, model, layer_name=args.layer_name, poison_rate=args.poison_ratio)  # layer4  conv_block3  avgpool
        ss.dump_stats(os.path.join('data', args.dataset, args.subset, 'SpectralSignature'), model.config, model_dir)
        # log
        name_cur = '_'.join([args.dataset, args.subset])
        result = [name_cur, args.layer_name] + res[1]

    elif name in ['strip']:
        # STRIP
        strip = STRIP(args.img_shape, args.num_classes, alpha=args.strip_alpha, N=args.strip_N, thres_fpr=args.strip_thres, device=device)
        print(strip.config)
        # res = strip.cleanse(model, ds_inspection=ds, ds_clean=ds_vals[0], batch_size=args.batch_size, poison_rate=args.poison_ratio)
        ds_val = merge_ds(ds_vals[1], ds_vals[0])
        res = strip.cleanse(model, ds_val=ds_val, ds_clean=ds, batch_size=args.batch_size, poison_rate=args.poison_ratio)
        # log
        name_cur = args.name if args.name is not None else '_'.join([args.dataset, args.subset_val])
        # name_cur = '_'.join([args.dataset, args.subset])
        result = [name_cur] + res[1] + [args.strip_alpha, args.strip_N, args.strip_thres, args.poison_ratio]

    elif name in ['scaleup', 'scale_up']:
        # Scale Up
        scaleup = ScaleUp(args.img_shape, args.num_classes, thres=args.scaleup_thres)
        print(scaleup.config)
        scaleup.init_spc_norm(ds, model, device)
        ds_val = merge_ds(ds_vals[1], ds_vals[0])
        res = scaleup.detect(ds_val, model, args.poison_ratio, device)
        # log
        name_cur = args.name if args.name is not None else '_'.join([args.dataset, args.subset_val])
        result = [name_cur] + res + [args.scaleup_thres, args.poison_ratio]


    elif name in ['nc', 'neural_clustering']:
        # Neural Cleanse
        nc = NeuralCleanse(args.img_shape, args.num_classes, tgt_ls=tgt_ls, pattern_per_target=1)  # [0, 1]
        print(nc.config)
        nc.reverse(model, ds, ds_vals, args.epochs, args.batch_size, args.num_workers, args.lr, 
                   reg_lambda=args.nc_reg_lambda, reg_name=args.nc_reg, patience=args.nc_patience, optimizer_name=args.optimizer, device=device)
        nc.outlier_detection()
        nc.show_one(0, 0, './', 'NCReverse_'+args.model_dir)
        # nc.dump_stats(os.path.join('res', args.model_dir, 'NCReverse'))  # TODO: restore this
        # log
        name_cur = args.model_dir
        result = [name_cur] + nc.config['mad_ls'] + nc.config['l1_ls'] + nc.config['asr'] + [args.epochs, args.lr]

    elif name in ['karm', 'k-arm', 'k_arm']:
        # K-Arm
        k_arm = KArm(args.img_shape, args.num_classes, )
        print(k_arm.config)
        if args.karm_prescreen:
            k_arm.pre_screening(model, ds, gamma=0, theta_general=0, theta_specific=0, device=device)
        else:
            k_arm.set_all_tgt(general=True, label_specific=True)
        # print(k_arm.config)  # [[15, 12]]
        k_arm.optimize(model, ds, tgt_ls=tgt_ls, steps_total=args.karm_steps, batch_size=args.batch_size, num_workers=args.num_workers, lr=args.lr, reg=args.karm_reg, device=device)  # [[7, 4]]
        res = k_arm.sym_check(model, ds, steps=args.karm_steps_sym, batch_size=args.batch_size, num_workers=args.num_workers, lr=args.lr, device=device)
        k_arm.dump_stats(os.path.join('res', args.model_dir, 'KArmReverse'))

    elif name in ['featurere', 'feature_re']:
        # FeatureRE
        featurere = FeatureRE(args.img_shape, args.num_classes, tgt_ls=tgt_ls)
        print(featurere.config)
        # featurere.reverse(model, ds, 0, device)
        res = featurere.reverse_all_tgt(model, ds, device)
        # log
        name_cur = args.model_dir
        result = [name_cur] + res['asr'] + res['mixedv'] + res['mixedv_best']

    elif name in ['abl', 'anti_backdoor_learning']:
        # Anti-Backdoor Learning
        abl = ABL(args.img_shape, args.num_classes, gamma=args.abl_gamma, device=device)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        output_dir = '_'.join(['ABL', args.dataset, args.subset])
        model = abl.train(ds, ds_vals, model, args.epochs, args.lr, criterion, args.optimizer, args.scheduler, args.batch_size, args.num_workers, output_dir)
        res = abl.filter(ds, model, criterion, args.abl_split_ratio, poison_rate=args.poison_ratio)
        # log
        name_cur = '_'.join([args.dataset, args.subset])
        result = [name_cur] + res[1] + [args.model, args.epochs, args.lr, args.abl_split_ratio, args.poison_ratio]




    else:
        raise Exception('Defense not implemented: ', args.defense)

    if args.log is not None and result is not None:
        print('log saved to ', args.log)
        result = ', '.join([str(i) for i in result])
        with open('logs/' + args.log, 'a') as f:
            f.write('\n' + result)
        return



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, help='GPU id', default='0')
    parser.add_argument('--seed', type=int, help='global seed', default=0)
    parser.add_argument('--log', type=str, help='path to log file', default=None)

    # dataset & model config
    parser.add_argument('--dataset', type=str, help='name of the dataset', default='cifar10')
    parser.add_argument('--ds_fmt', type=str, help='dataset format: pickle(pkl)/folder/offic', default='pkl')
    parser.add_argument('--num_classes', type=int, help='classes num of the dataset', default=None)
    parser.add_argument('--img_shape', type=list, nargs='+', default=None)
    parser.add_argument('--channel', type=int, help='channel num of imgs', default=None)
    parser.add_argument('--img_size', type=int, help='height/width of imgs', default=None)
    parser.add_argument('--subset', type=str, help='name of the dataset', default='clean')
    parser.add_argument('--subset_val', type=str, help='subset for validation', default=None)
    parser.add_argument('--split', type=str, help='train/test/val/dev', default='train')
    parser.add_argument('--split_val', type=str, help='train/test/val/dev', default='test')
    parser.add_argument('--split_val2', type=str, help='train/test/val/dev', default=None)

    parser.add_argument('--model', type=str, help='model name', default='resnet18')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, help='filename <best/last>', default='last')  # last/best

    parser.add_argument('--num_workers', '--nw', type=int, help='num of workers', default=4)
    parser.add_argument('--batch_size', '--bs', type=int, help='batch size', default=128)
    parser.add_argument('--epochs', type=int, help='num of epochs', default=40)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
    parser.add_argument('--optimizer', type=str, help='adam/sgd/adamdelta', default='sgd')
    parser.add_argument('--scheduler', type=str, help='step/plateau/cosann/multistep', default='cosann')

    # defense config
    parser.add_argument('--name', type=str, help='output_name', default=None)
    parser.add_argument('--target', type=int, help='attack target', default=None)
    parser.add_argument('--poison_ratio', type=float, help='backdoor inject ratio', default=None)
    parser.add_argument('--defense', type=str, help='defense name', default=None)

    parser.add_argument('--layer_name', type=str, help='layer to get feature in AC/SS', default=None)
    parser.add_argument('--decomposition', type=str, help='decomposition function in AC', default='FastICA')
    parser.add_argument('--strip_alpha', type=float, help='blend alpha in STRIP', default=0.5)
    parser.add_argument('--strip_N', type=int, help='sample num to blend in STRIP', default=64)
    parser.add_argument('--strip_thres', type=float, help='thres of FPR in STRIP', default=0.05)
    parser.add_argument('--scaleup_thres', type=float, help='thres in Scale Up', default=0.5)
    parser.add_argument('--nc_reg', type=str, help='regularization norm of NC', default='l1')
    parser.add_argument('--nc_reg_lambda', type=float, help='lambda of reg loss in NC', default=1e-1)
    parser.add_argument('--nc_patience', type=float, help='patience of changing reg lambda in NC', default=5)
    parser.add_argument('--karm_prescreen', action='store_true', help='use pre-screening in K-Arm')
    parser.add_argument('--karm_steps', type=int, help='total steps of K-Arm', default=1000)
    parser.add_argument('--karm_steps_sym', type=int, help='symmetric-check steps of K-Arm', default=40)
    parser.add_argument('--karm_reg', type=float, help='reg factor of loss in K-Arm', default=1.)
    # parser.add_argument('--abl_pre_epochs', type=int, help='pre-isolation training epochs in ABL', default=20)
    parser.add_argument('--abl_gamma', type=float, help='loss bound of pre-isolation in ABL', default=0.5)
    parser.add_argument('--abl_split_ratio', type=float, help='ratio of poisoned to be filtered in ABL', default=0.01)


    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    args.split_val2 = args.split_val if args.split_val2 is None else args.split_val2
    args.subset_val = args.subset if args.subset_val is None else args.subset_val
    main(args)
