'''
    Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks (https://ieeexplore.ieee.org/abstract/document/8835365)

    code：
        Backdoor Toolbox (https://github.com/vtu81/backdoor-toolbox/blob/main/other_defenses_tool_box/neural_cleanse.py)
'''
import os
import sys
import math
import pickle
from tqdm import tqdm

from PIL import Image
import numpy as np
import torch
import torchvision


class NeuralCleanse:
    def __init__(self, img_shape, num_classes, tgt_ls=None, pattern_per_target=1):
        self.img_shape = img_shape
        self.c, self.h, self.w = img_shape
        self.num_classes = num_classes
        self.tgt_ls = tgt_ls if tgt_ls else list(range(num_classes))
        self.ppt = pattern_per_target
        self.config = {'type': 'NeuralCleanse', 'targets': self.tgt_ls, 'img_shape': img_shape, 'num_classes': num_classes,
                       'pattern_per_target': self.ppt}

        # initialize trigger pattern & mask
        self.pattern_dict = {}
        for tgt in self.tgt_ls:
            cur_ls = []
            for _ in range(pattern_per_target):
                mask = torch.randn(self.img_shape[1:]).unsqueeze(0)
                pattern = torch.randn(self.img_shape)
                mask.requires_grad_()
                pattern.requires_grad_()

                cur_ls.append([mask, pattern])
            self.pattern_dict[tgt] = cur_ls

    def apply(self, x, tgt, idx, ):
        img = x[0]
        mask, pattern = self.pattern_dict[tgt][idx]
        mask_tanh, pattern_tanh = self.tanh(mask.clone().detach()), self.tanh(pattern.clone().detach())
        img = (img * (1 - mask_tanh) + mask_tanh * pattern_tanh).clamp(0., 1.)
        return img, tgt
    
    # for datapipe.map()
    def apply_fn(self, tgt, idx, normalizer=None, denormalizer=None):
        mask, pattern = self.pattern_dict[tgt][idx]
        mask_tanh, pattern_tanh = self.tanh(mask.clone().detach()), self.tanh(pattern.clone().detach())

        def fn(x):
            img = x[0]
            img = denormalizer(img) if denormalizer is not None else img
            img = (img * (1 - mask_tanh) + mask_tanh * pattern_tanh).clamp(0., 1.)
            img = normalizer(img) if normalizer is not None else img
            return img, tgt
        return fn
    
    def reverse(self, model, ds_gen, ds_vals=None, epochs=10, batch_size=32, num_workers=0, lr=0.1,
                reg_lambda=1e-1, reg_name='l1', reg_multiplier=1.5, optimizer_name='Adam', asr_thres=0.99, patience=5,
                early_stop=False, early_stop_thresh=0.99, mean=None, std=None, device='cpu'):
        from ...misc import get_regularization, get_optimizer, evaluate, Denormalize
        self.config.update({'epochs': epochs, 'batch_size': batch_size, 'num_workers': num_workers, 'lr': lr,
                            'reg_lambda': reg_lambda, 'regularization': reg_name, 'optimizer': optimizer_name})
        # freeze model params
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()

        # parse configs & initialize
        criterion = torch.nn.CrossEntropyLoss()
        reg = get_regularization(reg_name)
        mean = mean if mean is not None else [0.4802, 0.4481, 0.3975]
        std = std if std is not None else [0.2302, 0.2265, 0.2262]
        denormalizer = Denormalize(mean, std)
        normalizer = torchvision.transforms.Normalize(mean, std)
        # get dataloaders
        ds_vals = ds_vals if ds_vals else []
        dl_gen = torch.utils.data.DataLoader(ds_gen, batch_size, shuffle=False, num_workers=num_workers)
        asr_ls = []

        # reverse triggers of every tgt
        for tgt in self.tgt_ls:
            for idx in range(self.ppt):
                mask, pattern = self.pattern_dict[tgt][idx]
                optimizer = get_optimizer(optimizer_name, [mask, pattern], lr)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.4)
                reg_lambda_cur, counter_up, counter_down = reg_lambda, 0, 0
                asr_cur, l1_cur = 0, np.inf
                # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

                prog_bar = tqdm(range(epochs), file=sys.stdout, disable=False, leave=False)
                for epoch in prog_bar:
                    loss_epoch, counter_batch, num_acc, num_asr, num_total = 0, 0, 0, 0, 0
                    for step, (batch_x, batch_y) in enumerate(dl_gen):
                        mask_tanh, pattern_tanh = self.tanh(mask.to(device)), self.tanh(pattern.to(device))
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        batch_x = denormalizer(batch_x)
                        batch_x = normalizer(batch_x + mask_tanh * (pattern_tanh - batch_x).clamp(0., 1.))
                        batch_tgt = torch.tensor([tgt] * batch_x.shape[0], device=device)

                        optimizer.zero_grad()
                        pre = model(batch_x)  # .to(device)
                        loss = criterion(pre, batch_tgt) + reg_lambda_cur * reg(mask_tanh, torch.zeros((1, self.h, self.w), device=device))

                        loss.backward()
                        optimizer.step()
                        loss_epoch += loss.item()
                        counter_batch += 1

                        # calculate asr
                        pre = torch.max(pre.detach(), dim=1)[1]
                        num_acc += torch.eq(pre, batch_y).sum().item()
                        num_asr += torch.eq(pre, batch_tgt).sum().item()
                        num_total += batch_x.shape[0]
                        prog_bar.desc = "NC: reverse tgt[{}] epoch[{}/{}] loss:{:.4f} lr:{:.4f} lambda:{:.2f}  asr:{:.2f} l1:{:.2f}  " \
                            .format(tgt, epoch + 1, epochs, loss, optimizer.state_dict()['param_groups'][0]['lr'], reg_lambda_cur, asr_cur, l1_cur)
                    scheduler.step()

                    # modify lambda
                    l1_cur = torch.sum(torch.abs(self.tanh(mask.clone().detach()))).item()
                    acc_cur, asr_cur = num_acc/num_total, num_asr/num_total
                    if asr_cur > asr_thres:
                        counter_up, counter_down = counter_up + 1, 0
                    else:
                        counter_up, counter_down = 0, counter_down + 1
                    if counter_up >= patience:
                        counter_up, reg_lambda_cur = 0, reg_lambda_cur * reg_multiplier
                    elif counter_down >= patience:
                        counter_down, reg_lambda_cur = 0, reg_lambda_cur / reg_multiplier**1.5

                # val
                ds_val = ds_vals[1].map(self.apply_fn(tgt, idx, normalizer, denormalizer))
                dl_val = torch.utils.data.DataLoader(ds_val, batch_size, shuffle=False, num_workers=num_workers)
                loss_asr, asr, time_asr = evaluate(model, dl_val, criterion, device, batch_size, "asr", True)
                stat_cur = "tgt[{:>2}/{}] loss:{:.4f} elapsed:{:.2f}   loss:{:.6f} asr:{:.4f} l1:{:.2f}".format(
                    tgt, self.num_classes, loss_epoch / counter_batch, prog_bar.format_dict['elapsed'], loss_asr, asr, l1_cur)
                asr_ls.append(asr)
                print(stat_cur)
                # print()
                # self.visualize_one(mask.detach().numpy(), pattern.detach().numpy(), os.path.join("temp/", "{}_{}_E{}.png".format(tgt, idx, epoch)))
        self.config.update({'asr': asr_ls, })
        print('asr:', asr_ls)
        print('Neural Cleanse: reverse finished. ')

    def outlier_detection(self):
        c_const = 1.4826  # consistency_constant (if normal distribution)
        # get l1 norms
        l1_dict, l1_total_ls = {}, []
        for tgt in self.tgt_ls:
            cur_ls = []
            for idx in range(self.ppt):
                mask, pattern = self.pattern_dict[tgt][idx]
                l1_cur = torch.sum(torch.abs(self.tanh(mask.clone().detach()))).item()
                cur_ls.append(l1_cur)
            l1_dict[tgt] = cur_ls
            l1_total_ls.extend(cur_ls)
        print(l1_dict)

        # calculate median & median absolute deviation (MAD)
        median = np.median(l1_total_ls)
        mad = c_const * np.median(np.abs(np.array(l1_total_ls) - median))
        mad_min = np.abs(np.min(l1_total_ls) - median) / mad
        mad_ls = (np.abs(l1_total_ls - median)/mad).tolist()
        print('median: {},  MAD: {},  anomaly_index: {}'.format(median, mad, mad_min))

        # filter suspicious labels
        suspicious_tgt = []
        for tgt in self.tgt_ls:
            for idx in range(self.ppt):
                if l1_dict[tgt][idx] <= median and np.abs(l1_dict[tgt][idx] - median)/mad > 2:
                    if tgt not in suspicious_tgt:
                        suspicious_tgt.append(tgt)
                    print('suspicious_tgt: {}   idx: {}   l1_norm: {}'.format(tgt, idx, l1_dict[tgt][idx]))
        self.config.update({'suspicious_tgt': suspicious_tgt, 'l1_norm': l1_dict, 'l1_ls': l1_total_ls,
                            'median': median, 'MAD': mad, 'mad_ls': mad_ls, 'anomaly_index': mad_min})
        
        print('suspicious_tgt: ', suspicious_tgt)
        return suspicious_tgt

    def dump_stats(self, path, show=True, plot=True):
        import json
        os.makedirs(os.path.join(path), exist_ok=True)
        json.dump(self.config, open(os.path.join(path, 'stats.json'), 'w'), indent=4)
        print('Neural Cleanse stats saved to: ' + path)

        if show:
            self.show(path)
            if plot:
                pass

    def show_one(self, tgt, idx, path, verbose='', show=False):
        mask, pattern = self.pattern_dict[tgt][idx]
        mask, pattern = self.tanh(mask.clone().detach()).numpy(), self.tanh(pattern.clone().detach()).numpy()
        mask, pattern = mask.transpose(1, 2, 0), pattern.transpose(1, 2, 0)
        img = (np.dstack([pattern, mask]) * 255).squeeze().astype(np.uint8)
        # img = (mask * pattern * 255).squeeze().astype(np.uint8).transpose(1, 2, 0)
        img = Image.fromarray(img)
        img.save(os.path.join(path, '{}{}_{}.png'.format(verbose, tgt, idx)))
        if show:
            img.show()

    def show(self, path, verbose=''):
        os.makedirs(path, exist_ok=True)
        for tgt in self.tgt_ls:
            for idx in range(self.ppt):
                self.show_one(tgt, idx, path, verbose)
        print('NeuralCleanse: all pattern saved to {}'.format(path))

    @staticmethod
    def tanh(x: torch.Tensor) -> torch.Tensor:
        return (x.tanh() + 1) * 0.5

    @staticmethod
    def atan_func(x: torch.Tensor) -> torch.Tensor:
        return x.atan().div(math.pi).add(0.5)

    @staticmethod
    def tanh_np(x):
        return np.tanh(x) / 2 + 0.5

