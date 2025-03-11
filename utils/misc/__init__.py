import os
import sys
import re
import json
import random
import math
import time

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from copy import deepcopy
from PIL import ImageDraw, Image


from .metric import *
from .transforms import *
from .analysis import *
from .format import *


class Log:
    def __init__(self):
        self.loss_train, self.loss_acc, self.loss_asr = [], [], []
        self.time_train, self.time_acc, self.time_asr = [], [], []
        self.acc, self.asr = [], []
        self.stats = []
        self.best_acc = 0
        self.total_time = 0
        self.keys = ['loss_train', 'loss_acc', 'loss_asr', 'acc', 'asr', 'time_train', 'time_acc', 'time_asr']

    def dump(self, output_dir):
        # save loss/acc/time
        with open(output_dir, 'a+') as f:
            json.dump(self.get_dict())
        return
    
    def dump_csv(self, output_dir, epoch, vals):
        f_path = 'res/' + output_dir + '/results.csv'
        s = '' if os.path.exists(f_path) else (('%20s,' * (len(self.keys) + 1) % tuple(['epoch'] + self.keys)).rstrip(',') + '\n')  # add header
        with open(f_path, 'a+') as f:
            f.write(s + ('%20.5g,' * (len(self.keys) + 1) % tuple([epoch] + vals)).rstrip(',') + '\n')
        return
    
    def plot(self, output_dir, res_json=None, res_csv=None, plot_all=False):
        if res_csv is None:
            res_csv = 'res/' + output_dir + '/results.csv'
        fig, ax = plt.subplots(2, 4, figsize=(12, 6), tight_layout=True)
        ax = ax.ravel()
        data = np.array(list(self.get_dict().values())[:-1])
        x = list(range(len(data[0])))
        for i, j in enumerate([0, 1, 2, 3, 5, 6, 7, 4]):
            y = data[j]
            ax[i].plot(x, y, marker='.', label=self.keys[j], linewidth=2, markersize=8)
            ax[i].set_title(self.keys[j], fontsize=12)
            # if j in [8, 9, 10]:  # share train and val loss y axes
            #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        fig.savefig('res/'+output_dir+'/results.png', dpi=200)
        plt.close()

        if plot_all:
            self.plot_loss(output_dir)
            self.plot_time(output_dir)
            self.plot_acc(output_dir)
        return
    
    def plot_loss(self, output_dir, data=None):
        if data is None:
            data = [self.loss_train, self.loss_acc, self.loss_asr]
        labels = ['loss_train', 'loss_acc', 'loss_asr']
        plt.figure()
        plt.title('loss')
        for i in range(len(data)):
            plt.plot(range(len(data[0])), data[i], label=labels[i])
        plt.legend()
        plt.grid()
        plt.savefig('res/' + output_dir + '/loss.png')
        plt.close()

    def plot_acc(self, output_dir, data=None):
        if data is None:
            data = [self.acc, self.asr]
        labels = ['acc', 'asr']
        plt.figure()
        plt.title('acc/asr')
        for i in range(len(data)):
            plt.plot(range(len(data[0])), data[i], label=labels[i])
        plt.legend()
        plt.grid()
        plt.savefig('res/' + output_dir + '/acc.png')
        plt.close()
    
    def plot_time(self, output_dir, data=None):
        if data is None:
            data = [self.time_train, self.time_acc, self.time_asr]
        labels = ['time_train', 'time_acc', 'time_asr']
        plt.figure()
        plt.title('time')
        for i in range(len(data)):
            plt.plot(range(len(data[0])), data[i], label=labels[i])
        plt.legend()
        plt.grid()
        plt.savefig('res/' + output_dir + '/time.png')
        plt.close()

    def get_dict(self):
        return dict(loss_train=self.loss_train, loss_acc=self.loss_acc, loss_asr=self.loss_asr,
                    acc=self.acc, asr=self.asr,
                    time_train=self.time_train, time_acc=self.time_acc, time_asr=self.time_asr,
                    stats=self.stats)
    
    def read_json(self, path='a_test'):
        with open('res/'+path+'/stats.json', 'r') as f:
            data = json.load(f)['training_stats']
        self.loss_train, self.loss_acc, self.loss_asr, self.acc, self.asr, self.time_train, self.time_acc, self.time_asr = data["loss_train"], data["loss_acc"], data["loss_asr"], data["acc"], data["asr"], data["time_train"], data["time_acc"], data["time_asr"]
        # self.__dict__ = data
        self.loss_train = [self.loss_train[2*i] for i in range(int(len(self.loss_train)/2))]
        return 
    

def get_device(gpu_idx=None):
    if gpu_idx in [None, "cpu"]:
        return torch.device("cpu")
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:"+str(gpu_idx) if torch.cuda.is_available() else "cpu")
    print("using device {} ".format(device))
    return device


def set_hf_cache():
    os.environ['TRANSFORMERS_CACHE'] = './hfcache/'


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def fix_multiprocessing():
    # fix cpu scheduling of torch multiprocessing (when import mkl)
    import torch.multiprocessing as tmultiprocessing
    import time

    class MyProcessT(tmultiprocessing.Process):
        def __init__(self):
            super(MyProcessT, self).__init__()

        def run(self):
            time.sleep(1)
            # print('hello', self.name, time.ctime())

    w_temp = MyProcessT()
    w_temp.start()
    

def get_nw(default=8, batch_size=8):
    if os.name == 'nt':
        print("OS: Windows")
        return 0
    elif os.name == 'posix':
        print("OS: Linux/Unix/Mac")
        return min([os.cpu_count(), batch_size if batch_size > 1 else 0, default])
    else:
        raise Exception("Unknown OS: ", os.name)


def get_optimizer(optimizer_type, params, lr):
    from torch.optim import Adam, SGD, Adadelta
    if optimizer_type.lower() in ['sgd']:
        return SGD(params, lr, weight_decay=5e-4, momentum=0.9)  # momentum=0.9
    elif optimizer_type.lower() in ['adam']:
        return Adam(params, lr, weight_decay=1e-5)  # 1e-5
    elif optimizer_type.lower() in ['adadelta']:
        return Adadelta(params, lr)
    else:
        raise Exception("Optimizer not implemented")


def get_regularization(regularization=None):
    from torch.nn import L1Loss, MSELoss
    if regularization is None or regularization in ['None', 'none']:
        return lambda arg0, arg2: 0 
    elif regularization in ['l1', 'L1']:
        return L1Loss()
    elif regularization in ['l2', 'L2']:
        return MSELoss()
    else:
        raise Exception("Regularization not implemented")


def get_scheduler(scheduler, optimizer, epochs=0, lr_step=10, lr_factor=0.1, lr_patience=5, lr_min=0.5e-6, milestones=[], gamma=0.1):
    if scheduler.lower() in ['step', 'steplr']:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_factor)
    elif scheduler.lower() in ['plateau']:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_factor, cooldown=0, patience=lr_patience, min_lr=lr_min)
    elif scheduler.lower() in ['cosineannealing', 'cosann']:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler.lower() in ['multistep', 'multisteplr']:
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    else:
        raise Exception("Scheduler not implemented: ", scheduler)
    
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)



def timer(func):
    def wrapper(*args, **kwargs):
        time_start = time.perf_counter()
        func(*args, **kwargs)
        print("elapsed: ", time.perf_counter() - time_start)

    return wrapper


def no_grad(func):
    def wrapper_nograd(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return wrapper_nograd


def iou_np(boxes1, boxes2):
    """
    cal IoU for ndarray boxes(y1, x1, y2, x2): (N,4),(M,4) -> (N, M)
    """
    top_left_point = np.maximum(boxes1[:, None, 0:2], boxes2[:, 0:2])  # (N,1,2),(M,2) -> (N,M,2) indicating top-left corners of box pairs
    bottom_right_point = np.minimum(boxes1[:, None, 2:4], boxes2[:, 2:4])  # bottom-right corners
    well_ordered_mask = np.all(top_left_point < bottom_right_point, axis=2)  # (N,M) indicating whether top_left_x < bottom_right_x and top_left_y < bottom_right_y (meaning boxes may intersect)
    intersection_areas = well_ordered_mask * np.prod(bottom_right_point - top_left_point, axis=2)  # (N,M) indicating intersection area (bottom_right_x - top_left_x) * (bottom_right_y - top_left_y)
    areas1 = np.prod(boxes1[:, 2:4] - boxes1[:, 0:2], axis=1)  # (N,) areas of boxes1
    areas2 = np.prod(boxes2[:, 2:4] - boxes2[:, 0:2], axis=1)  # (M,) areas of boxes2
    union_areas = areas1[:, None] + areas2 - intersection_areas  # (N,1) + (M,) - (N,M) = (N,M)
    return intersection_areas / (union_areas + 1e-7)


def deltas_to_boxes(deltas, anchors, means=None, stds=None, device=None):
    """
    input:(tensor, tensor)
        box_deltas[bs, N, 4(ty, tx, th, tw)], anchors[bs, 4(centery, centerx, h, w)]
    output:(tensor)
        boxes[bs, N, 4(y1, x1, y2, x2)]
    # center_x = anchor_width * tx + anchor_center_x, center_y = anchor_height * ty + anchor_center_y
    # width = anchor_width * exp(tw), height = anchor_height * exp(th)
    """
    if means is None:
        means = [0., 0., 0., 0.]
    if stds is None:
        stds = [1., 1., 1., 1.]
    means, stds = torch.tensor(means).to(device), torch.tensor(stds).to(device)
    deltas, anchors = deltas.to(device), anchors.to(device)
    deltas = deltas * stds + means  # normalize

    center = anchors[:, 2:4] * deltas[:, 0:2] + anchors[:, 0:2]
    size = anchors[:, 2:4] * torch.exp(deltas[:, 2:4])
    boxes = torch.empty(deltas.shape, dtype=torch.float32, device=device)
    boxes[:, 0:2] = center - 0.5 * size  # y1, x1
    boxes[:, 2:4] = center + 0.5 * size  # y2, x2
    return boxes



def draw_box(image, boxes, color=(256, 256, 256), width=2):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        y1, x1, y2, x2 = box[1]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
    return image


def fit(model, epochs, criterion, optimizer, dl_train, dl_vals, scheduler, device, batch_size=32,
        output_dir=None, save_last=True, return_log=False, plot=False, disable_prog=False):
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
            optimizer.zero_grad()
            pre = model(batch_x.to(device))
            loss = criterion(pre, batch_y.to(device))

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
        loss_acc, acc, time_acc = evaluate(model, dl_vals[0], criterion, device, batch_size, "acc", disable_prog)
        loss_asr, asr, time_asr = evaluate(model, dl_vals[1], criterion, device, batch_size, "asr", disable_prog)

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


def evaluate(model, dl_val, criterion, device, batch_size=32, verbose="acc", disable=False, output_info=False):
    model.eval()
    num_total, num_acc = 0, 0
    prog_bar = tqdm(dl_val, file=sys.stdout, leave=False, disable=disable)
    time_start = time.perf_counter()
    with torch.no_grad():
        for _, (batch_x, batch_y) in enumerate(prog_bar, 0):
            pre = model(batch_x.to(device))
            loss = criterion(pre, batch_y.to(device)).item()
            pre = torch.max(pre, dim=1)[1]
            # pre = torch.round(pre)  # BCELoss
            num_acc += torch.eq(pre, batch_y.to(device)).sum().item()
            num_total += batch_y.size(0)
            prog_bar.desc = "valid[{}] ".format(verbose)
    acc = num_acc / num_total
    elapsed = time.perf_counter() - time_start  # prog_bar.format_dict['elapsed']
    if output_info:
        print('loss: %.6f  %s: %.4f (%d/%d)' % (loss, verbose, acc, num_acc, num_total))
    return loss, acc, elapsed
