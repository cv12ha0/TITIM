'''
    Backdoor Scanning for Deep Neural Networks through K-Arm Optimization (https://proceedings.mlr.press/v139/shen21c.html)

    code:
        official (https://github.com/PurduePAML/K-ARM_Backdoor_Optimization)
'''
import os
import sys
import math
import random
import pickle
from tqdm import tqdm

from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, Subset

from utils.misc import get_regularization, get_optimizer
from utils.datasets import get_indices

__all__ = [
    "KArm", 
]


class KArm:
    def __init__(self, img_shape, num_classes, ):
        self.img_shape = img_shape
        self.c, self.h, self.w = img_shape
        self.num_classes = num_classes
        self.tgt_ls = []  # [(None, target), (victim, target), ...]

        self.config = {'type': 'K-Arm', 'targets': self.tgt_ls, 'img_shape': img_shape, 'num_classes': num_classes, }

        # {(victim, target): [pattern, mask, optimizer, dl; reg, times, best_mask_norm, acc; victim, target; counters]}
        self.status_dict = {}
        self.suspicous_pairs = None
        pass

    def set_all_tgt(self, general=True, label_specific=True):
        # general
        if general:
            self.tgt_ls.extend([[None, i] for i in range(self.num_classes)])
        # label-specific
        if label_specific:
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    if i == j:
                        continue
                    self.tgt_ls.append([i, j])
        self.config['targets'] = self.tgt_ls
        print('K-Arm: using all pairs. \n')

    def pre_screening(self, model, ds, gamma=0.25, theta_general=0.95, theta_specific=0.9, batch_size=32, device='cpu'):
        self.config.update({'gamma': gamma, 'theta_general': theta_general, 'theta_specific': theta_specific})
        # gen logits for ranking
        model.eval()
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
        prog_bar = tqdm(dl, file=sys.stdout, disable=False)
        logits, preds = torch.tensor([]), torch.tensor([])
        with torch.no_grad():
            for _, (batch_x, batch_y) in enumerate(prog_bar, 0):
                output = model(batch_x.to(device))
                pred = torch.max(output, dim=1)[1]
                logits = torch.cat((logits, output.detach().cpu()), 0)
                preds = torch.cat((preds, pred.detach().cpu()), 0)

        # sort logits in desc & reserve topk
        k = max(2, round(self.num_classes * gamma))
        topk_logit, topk_label = torch.topk(logits, k, dim=1)

        # check general trigger
        counter = torch.zeros(self.num_classes)
        for i in range(self.num_classes):
            counter[i] = topk_label[topk_label == i].size(0)
        if torch.max(counter) > theta_general * topk_label.size(0):
            self.tgt_ls.append([None, torch.argmax(counter).item()])
            # return torch.argmax(counter)

        # check label-specific trigger
        sum_mat = torch.zeros(self.num_classes, self.num_classes)
        median_mat = torch.zeros(self.num_classes, self.num_classes)
        for i in range(self.num_classes):
            # for every victim, filter the corresponding logits
            topk_label_cur = topk_label[topk_label[:, 0] == i]
            topk_logit_cur = topk_logit[topk_label[:, 0] == i]

            # check each pair of labels
            for j in range(self.num_classes):
                if i == j:
                    continue
                if topk_label_cur[topk_label_cur == j].size(0) >= theta_specific * topk_label_cur.size(0):
                    sum_mat[j, i] = topk_logit_cur[topk_label_cur == j].sum()
                    median_mat[j, i] = torch.median(topk_logit_cur[topk_label_cur == j])

        targets, victims = [], []
        # for each target
        for i in range(self.num_classes):
            if sum_mat[i].max() > 0:
                victims_cur = sum_mat[i].nonzero().view(-1)
                victims_cur_sum = sum_mat[i][sum_mat[i] > 0]
                victims_cur_median = median_mat[i][sum_mat[i] > 0]

                # take intersection of two matrices
                victims_cur_sum_top = (victims_cur_sum > 1e-8).nonzero()[:, 0]
                victims_cur_median_top = (victims_cur_median > 1e-8).nonzero()[:, 0]
                cur_top_label = torch.LongTensor(np.intersect1d(victims_cur_sum_top, victims_cur_median_top))

                if len(cur_top_label) > 0:
                    victims_cur = victims_cur[cur_top_label]
                    if victims_cur.size(0) > 3:
                        cur_top_label = torch.topk(victims_cur_sum[cur_top_label], 3, dim=0)[1]
                        victims_cur = victims_cur[cur_top_label]

                    self.tgt_ls.extend([(victim.item(), i) for victim in victims_cur])
                    targets.append(i)
                    victims.append(victims_cur.tolist())
        self.config['targets'] = self.tgt_ls
        print('K-Arm Pre Screening: targets set to ', self.tgt_ls, '\n')
        print('K-Arm: pre_screening finished. \n')
        return targets, victims

    def optimize_step(self, model, victim, target, pattern, mask, optimizer, dl, reg,
                      regularization, criterion, epochs, device='cpu', verbose=''):
        acc, mask_norm = None, None
        for epoch in range(epochs):
            loss_all, num_acc, num_total, mask_norm_ls = 0, 0, 0, []
            prog_bar = tqdm(dl, file=sys.stdout, disable=True, leave=False)
            for batch_x, _ in prog_bar:
                pattern_tanh = torch.tanh(pattern) / (2 - 1e-7) + 0.5
                mask_tanh = torch.tanh(mask) / (2 - 1e-7) + 0.5
                batch_x = batch_x.to(device) * (1 - mask_tanh) + mask_tanh * pattern_tanh
                batch_y = torch.tensor([target] * batch_x.shape[0])

                optimizer.zero_grad()
                pre = model(batch_x.to(device))
                mask_norm = regularization(mask_tanh, torch.zeros((1, self.h, self.w)).to(device))
                loss = criterion(pre, batch_y.to(device)) + reg * mask_norm

                loss.backward()
                optimizer.step()
                loss_all += loss.item()
                num_acc += torch.eq(torch.max(pre, dim=1)[1], batch_y.to(device)).sum().item()
                num_total += batch_y.size(0)
                mask_norm_ls.append(mask_norm.item())
                prog_bar.desc = "K-Arm Optim {}tgt:({}, {}) epoch[{}/{}] loss:{:.4f} lr:{:.6f}" \
                    .format(verbose, victim, target, epoch + 1, epochs, loss, optimizer.state_dict()['param_groups'][0]['lr'])
            acc = num_acc / num_total  # ASR
            mask_norm = np.sum(np.abs(np.tanh(mask.cpu().detach().numpy()) / 2 + 0.5)).astype(float)
        return acc, mask_norm

    def optimize_step_arm(self, model, arm, regularization, criterion, epochs=1, device='cpu'):
        arm['times'] += 1
        pattern, mask, optimizer, dl, reg, times, best_mask_norm, acc, victim, target, *_ = arm.values()
        acc, mask_norm = self.optimize_step(model, victim, target, pattern, mask, optimizer, dl, reg, regularization, criterion, epochs, device)

        arm['acc'], arm['best_mask_norm'] = acc, mask_norm
        arm['optim_history'].append('{}-{} t{:<3}  acc{:.3f}  l1_norm{:.6f}  reg{:.4f}  '.format(victim, target, times, acc, mask_norm, reg))
        return acc, mask_norm

    def optimize(self, model, ds, tgt_ls=None, steps_total=1000, epochs=1, batch_size=32, num_workers=0, lr=1e-1,
                 regularization_name='l1', optimizer_name='adam', warm_up=2, early_stop=None, es_patience=10,
                 reg=1e-1, asr_thres=0.95, epsilon=0.3, beta=1e+4, central_init=False, device='cpu'):
        self.tgt_ls = tgt_ls if tgt_ls else self.tgt_ls
        self.config.update({'steps_total': steps_total, 'batch_size': batch_size, 'num_workers': num_workers, 'lr': lr,
                            'regularization': regularization_name, 'optimizer': optimizer_name, 'warm_up': warm_up,
                            'early_stop': early_stop, 'es_patience': es_patience, 'asr_thres': asr_thres,
                            'epsilon': epsilon, 'beta': beta, })

        # freeze model params
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()

        criterion = torch.nn.CrossEntropyLoss()
        regularization = get_regularization(regularization_name)

        # initialize patterns
        for victim, target in self.tgt_ls:
            pattern_cur, mask_cur = torch.rand(self.img_shape).to(device), torch.rand((1, self.h, self.w)).to(device)
            pattern_cur.requires_grad_(True)
            mask_cur.requires_grad_(True)
            if central_init:
                mask_cur *= 0.001
                mask_cur[:, int(self.h / 3): int(self.h * 2 / 3), int(self.h / 3): int(self.h * 2 / 3)] = 0.99
                mask_cur = torch.clamp(mask_cur, 0, 1)

            optimizer_cur = get_optimizer(optimizer_name, (mask_cur, pattern_cur), lr)

            indices_victim = get_indices(ds, victim, self.num_classes)
            ds_cur = Subset(ds, indices_victim)
            dl_cur = DataLoader(ds_cur, batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
            self.status_dict[victim, target] = {'pattern': pattern_cur, 'mask': mask_cur, 'optimizer': optimizer_cur,
                                                'dl': dl_cur, 'reg': reg, 'times': 0, 'best_mask_norm': 1e+10, 'acc': 0,
                                                'victim': victim, 'target': target,
                                                'es_cnt': 0, 'asr_asc_cnt': 0, 'asr_desc_cnt': 0,
                                                'optim_history': []}  # [times, acc, mask_norm, reg]
        print('patterns initialized. ')

        # warm up
        prog_bar = tqdm(self.tgt_ls, file=sys.stdout, disable=False, leave=True, desc='K-Arm warm-up: ')
        for victim, target in prog_bar:
            prog_bar.set_postfix_str(" arm_cur: [{} - {}] ".format(victim, target))
            arm_cur = self.status_dict[victim, target]
            # pattern, mask, optimizer, dl, reg, times, es_cnt, best_mask_norm, *_ = arm_cur
            for step in range(warm_up):
                acc, mask_norm = self.optimize_step_arm(model, arm_cur, regularization, criterion, epochs, device)
                # self.status_dict[victim, target].update({'best_mask_norm': mask_norm, 'acc': acc})
        print('warm up done. ')

        # optimize one arm per step
        prog_steps = tqdm(range(steps_total), file=sys.stdout, disable=False, leave=True, desc='K-Arm optimize: ')
        for step in prog_steps:
            # select an arm (k-arm scheduler)
            if random.random() > epsilon:
                arm_ls, score_ls = [], []
                for key in self.status_dict:
                    mask_norm, time = self.status_dict[key]['best_mask_norm'], self.status_dict[key]['times']
                    arm_ls.append(key)
                    score_ls.append((2500 - mask_norm)/time + beta / mask_norm)
                victim, target = arm_ls[np.argmax(score_ls)]
                # victim, target = arm_ls[np.argmax((2500 - np.array(mask_norms))/np.array(times) + self.beta / np.array(mask_norms))]
            else:
                victim, target = random.choice(list(self.status_dict.keys()))

            # optimize current arm
            arm_cur = self.status_dict[victim, target]
            pattern, mask, optimizer, dl, reg, times, best_mask_norm, *_ = arm_cur.values()
            acc, mask_norm = self.optimize_step_arm(model, arm_cur, regularization, criterion, epochs, device)

            # early stop
            if early_stop:
                # # set early stop counter
                # if mask_norm > best_mask_norm:
                #     arm_cur['es_cnt'] += 1
                # else:
                #     arm_cur['es_cnt'] = 0
                #     break
                pass

            # set reg, change weight of asr and mask_norm (alpha in objective function)
            if acc >= asr_thres:
                arm_cur['asr_asc_cnt'] += 1
                arm_cur['asr_desc_cnt'] = 0
                if arm_cur['asr_asc_cnt'] > 5:
                    arm_cur['reg'] *= 1.5
                    arm_cur['asr_asc_cnt'] = 0
            if acc < asr_thres:
                arm_cur['asr_asc_cnt'] = 0
                arm_cur['asr_desc_cnt'] += 1
                if arm_cur['asr_desc_cnt'] > 5:
                    arm_cur['reg'] /= 1.5
                    arm_cur['asr_desc_cnt'] = 0

            prog_steps.set_postfix_str(" arm_cur: [{} - {}] ".format(victim, target))

        print('K-Arm: optimization finished. \n')

    def sym_check(self, model, ds, sym_thres=10, steps=40, batch_size=32, num_workers=0, epochs=1, lr=1e-1,
                  regularization_name='l1', optimizer_name='adam', reg_init=1e-1, asr_thres=0.95, device='cpu'):
        # freeze model params
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()

        criterion = torch.nn.CrossEntropyLoss()
        regularization = get_regularization(regularization_name)

        # symmetric optimization for every label-specific pairs
        prog_pairs = tqdm(self.tgt_ls, file=sys.stdout, disable=False, leave=True, desc='K-Arm optimize-sym: ')
        for victim, target in prog_pairs:
            # pass general
            if victim is None:
                continue
            # temp arm initialize
            pattern, mask = torch.rand(self.img_shape).to(device), torch.rand((1, self.h, self.w)).to(device)
            pattern.requires_grad_(True)
            mask.requires_grad_(True)

            optimizer = get_optimizer(optimizer_name, (pattern, mask), lr)
            indices_victim = get_indices(ds, target, self.num_classes)
            ds_cur = Subset(ds, indices_victim)
            dl = DataLoader(ds_cur, batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
            reg, asr_asc_cnt, asr_desc_cnt, optim_history_sym = reg_init, 0, 0, []

            prog_steps = tqdm(range(steps), file=sys.stdout, disable=True, leave=True, desc='K-Arm optimize-sym: ')
            for step in prog_steps:
                acc, mask_norm = self.optimize_step(model, victim, target, pattern, mask, optimizer, dl, reg, regularization, criterion, epochs, device)
                optim_history_sym.append('{}-{}  acc{:.3f}  l1_norm{:.6f}  reg{:.4f}  '.format(victim, target, acc, mask_norm, reg))

                # set reg
                if acc >= asr_thres:
                    asr_asc_cnt += 1
                    asr_desc_cnt = 0
                    if asr_asc_cnt > 5:
                        reg *= 1.5
                        asr_asc_cnt = 0
                if acc < asr_thres:
                    asr_asc_cnt = 0
                    asr_desc_cnt += 1
                    if asr_desc_cnt > 5:
                        reg /= 1.5
                        asr_desc_cnt = 0
            self.status_dict[victim, target].update({'pattern_sym': pattern, 'mask_sym': mask, 'mask_norm_sym': mask_norm,
                                                     'acc_sym': acc, 'optim_history_sym': optim_history_sym})

        # symmetric check (mask l1 norm)
        suspicous_pairs = []
        for victim, target in self.tgt_ls:
            if not victim:
                suspicous_pairs.append([victim, target])
            else:
                mask_norm = self.status_dict[victim, target]['best_mask_norm']
                mask_norm_sym = self.status_dict[victim, target]['mask_norm_sym']
                if mask_norm_sym > mask_norm * sym_thres:
                    suspicous_pairs.append([victim, target])
        self.suspicous_pairs = suspicous_pairs
        print('K-Arm: symmetric check finished. \n')
        print('Suspicious pairs: ', suspicous_pairs)
        return suspicous_pairs

    '''
        dump & visualization
    '''
    def dump_stats(self, path, sym=True, show=True, plot=True):
        import json
        os.makedirs(os.path.join(path), exist_ok=True)
        info, optim_hist = {}, {}
        # optimization history
        for v, t in self.tgt_ls:
            optim_hist.update({'{}-{}'.format(v, t): {'optim': self.status_dict[v, t]['optim_history']}})
            if sym and v is not None:
                optim_hist.update({'{}-{}'.format(v, t): {'sym_optim': self.status_dict[v, t]['optim_history_sym']}})
        json.dump(optim_hist, open(os.path.join(path, 'optim_history.json'), 'w'), indent=4)
        # info
        summary = []
        for v, t in self.tgt_ls:
            arm_cur = self.status_dict[v, t]
            info.update({'{}-{}'.format(v, t): {'times': arm_cur['times'], 'asr': arm_cur['acc'], 'l1_norm': arm_cur['best_mask_norm']}})
            str_cur = '{}-{}  times:{}  asr:{}  l1_norm:{}'.format(v, t, arm_cur['times'], arm_cur['acc'], arm_cur['best_mask_norm'])
            if sym and v is not None:
                info.update({'{}-{}'.format(v, t): {'asr_sym': arm_cur['acc_sym'], 'l1_norm_sym': arm_cur['mask_norm_sym'], }})
                str_cur += '  asr_sym:{}  l1_norm_sym:{}'.format(arm_cur['acc_sym'], arm_cur['mask_norm_sym'])
            summary.append(str_cur)
        info.update({'summary': summary, 'suspicous_pairs': self.suspicous_pairs, })
        json.dump(info, open(os.path.join(path, 'stats.json'), 'w'), indent=4)
        print('K-Arm optim stats saved to: ' + path)

        if show:
            self.show(path, sym)
        if plot:
            self.plot_stats(path, sym)

    def show_one(self, img, path, fname='', show=False):
        img = img.cpu().detach().numpy().transpose(1, 2, 0).squeeze()
        img = Image.fromarray(np.uint8(img * 255))  # img.astype(np.uint8)
        img.save(os.path.join(path, fname))
        if show:
            img.show()

    def show(self, path, show_sym=True, verbose=''):
        path = os.path.join(path, 'patterns')
        os.makedirs(path, exist_ok=True)
        for victim, target in self.status_dict:
            acc = self.status_dict[victim, target]['acc']
            times = self.status_dict[victim, target]['times']
            pattern, mask = self.status_dict[victim, target]['pattern'], self.status_dict[victim, target]['mask']
            fname = '{}{}_{}_asr{}_t{}.png'.format(verbose, victim, target, acc, times, )
            self.show_one((self.tanh(mask) * self.tanh(pattern)), path, fname)
            fname = '{}{}_{}_mask_asr{}_t{}.png'.format(verbose, victim, target, acc, times, )
            self.show_one(self.tanh(mask), path, fname)

            if victim is not None and show_sym:
                acc = self.status_dict[victim, target]['acc_sym']
                pattern, mask = self.status_dict[victim, target]['pattern_sym'], self.status_dict[victim, target]['mask_sym']
                fname = '{}{}_{}_sym_asr{}.png'.format(verbose, victim, target, acc, )
                self.show_one((self.tanh(mask) * self.tanh(pattern)), path, fname)
                fname = '{}{}_{}_sym_mask_asr{}.png'.format(verbose, victim, target, acc, )
                self.show_one(self.tanh(mask), path, fname)
        print('K-Arm: all pattern saved to {}'.format(path))

    def plot_stats(self, path, sym=True, show=True):
        import matplotlib.pyplot as plt
        import matplotlib.pylab as pylab
        import seaborn as sns
        import pandas as pd
        np.set_printoptions(precision=2, suppress=True)
        # pd.options.display.float_format = '{:.2f}'.format
        pd.set_option('display.float_format', lambda x: '%2.f' % x)
        params = {
            "font.size": 12,
            'axes.labelsize': '35',  # 轴上字
            'xtick.labelsize': '27',  # 轴图例
            'ytick.labelsize': '27',  # 轴图例
            'lines.linewidth': 2,  # 线宽
            'legend.fontsize': '27',  # 图例大小
            # 'figure.figsize': '16, 9'  # set figure size,长12，宽9
        }
        pylab.rcParams.update(params)
        # plt.figure(figsize=(self.num_classes * 4 / 3, self.num_classes))

        def plot(key, fname, use_g=True, dtype=np.float32):
            arr_g = np.zeros(self.num_classes, dtype)
            arr = np.zeros((self.num_classes, self.num_classes), dtype)
            for victim, target in self.tgt_ls:
                arm_cur = self.status_dict[victim, target]
                if victim is None:
                    if use_g:
                        arr_g[target] = arm_cur[key]
                else:
                    arr[victim][target] = arm_cur[key]

            data = np.concatenate([arr, [arr_g]]) if use_g else arr
            index = (list(range(self.num_classes)) + ['None']) if use_g else list(range(self.num_classes))
            ax = sns.heatmap(pd.DataFrame(data, columns=range(self.num_classes), index=index),
                             annot=True, annot_kws={'size': 6},  # cbar_kws={'label': 'opt steps'},
                             xticklabels=True, yticklabels=True, square=True, cmap="YlGnBu")
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=12)
            plt.xlabel('target', fontsize=16)  # , fontsize=12
            plt.ylabel('victim', fontsize=16)
            plt.tick_params(labelsize=10)
            plt.title(fname, fontsize=18)
            # plt.xticks([0.1, 1.1, 2.1], ["mnist", "cifar", "gtsrb"], color='black', rotation=0)
            plt.savefig(os.path.join(path, fname + '.jpg'), dpi=1000, bbox_inches='tight')  # 'mnist_' + xlabel + '.png'
            if show:
                plt.show()

        os.makedirs(os.path.join(path), exist_ok=True)
        plot('times', 'opt_times', True, dtype=np.uint8)
        plot('acc', 'asr', True)
        plot('best_mask_norm', 'l1_norm', True)
        if sym:
            plot('mask_norm_sym', 'l1_norm_sym', False)
            plot('acc_sym', 'asr_sym', False)

    '''
        utils
    '''
    @staticmethod
    def tanh(x: torch.Tensor) -> torch.Tensor:
        return x.tanh().add(1).mul(0.5)

    @staticmethod
    def atan_func(x: torch.Tensor) -> torch.Tensor:
        return x.atan().div(math.pi).add(0.5)

    @staticmethod
    def tanh_np(x):
        return np.tanh(x) / 2 + 0.5
        


    '''
        deprecated
    '''
    def get_opt_times(self):
        opt_g = np.zeros(self.num_classes, np.uint8)
        opt = np.zeros((self.num_classes, self.num_classes), np.uint8)
        for victim, target in self.tgt_ls:
            arm_cur = self.status_dict[victim, target]
            if victim is None:
                opt_g[target] = arm_cur['times']
            else:
                opt[victim][target] = arm_cur['times']
        print(opt)
        # print(opt.tolist())
        print(opt_g)
        # print(opt_g.tolist())


# class ClsSampler(Sampler):
#     def __init__(self, victim):
#         self.victim = victim
#         self.mask = [0, 1, 2, 3, 4, 5]

#     def __iter__(self):
#         # temp = (self.indices[i] for i in torch.nonzero(self.mask))
#         return [0, 1, 2, 3, 4, 5]

#     def __len__(self):
#         return len(self.mask)

