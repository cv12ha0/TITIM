'''
    Rethinking the Reverse-engineering of Trojan Triggers (https://proceedings.neurips.cc/paper_files/paper/2022/hash/3f9bf45ea04c98ad7cb857f951f499e2-Abstract-Conference.html)

    code:
        Official (https://github.com/RU-System-Software-and-Security/FeatureRE)
        Backdoor Toolbox (https://github.com/vtu81/backdoor-toolbox/blob/main/other_defenses_tool_box/feature_re.py)
'''
import os
import sys
import warnings
import copy
import math
from tqdm import tqdm
from sklearn import metrics

import numpy as np
import torch
import torchvision

from utils.misc import CompactJSONEncoder, Denormalize, get_features, get_features2, timer
from utils.models import UNet
from functools import reduce

__all__ = [
    'FeatureRE', 
]


class FeatureRE:
    def __init__(self, img_shape, num_classes, tgt_ls=None, epochs=400, epochs_wp=100, epochs_mask=1, layer_name='layer4', device='cpu'):
        self.img_shape = img_shape
        self.c, self.h, self.w = img_shape
        self.num_classes = num_classes
        self.tgt_ls = tgt_ls if tgt_ls else list(range(num_classes))
        self.device = device

        self.epochs = epochs
        self.epochs_wp = epochs_wp  # warm up
        self.epochs_mask = epochs_mask  # optimize mask k times / epoch
        self._EPSILON = 1e-7
        self.layer_name = layer_name


        self.config = {'type': 'FeatureRE', 'img_shape': img_shape, 'num_classes': num_classes, 'tgt_ls': self.tgt_ls,
                       'epochs': self.epochs, 'epochs_wp': self.epochs_wp, 'layer_name': self.layer_name}

    def reverse_all_tgt(self, model, ds, device='cpu'):
        res = {'asr': [], 'loss_ce': [], 'loss_dist': [], 'loss_p': [], 'loss_mask': [], 'loss_std': [], 'mixedv': [], 'mixedv_best': []}
        for target in self.tgt_ls:
            _, asr, mixedv, mixedv_best = self.reverse(model, ds, target, device)
            res['asr'].append(asr)
            res['mixedv'].append(mixedv)
            res['mixedv_best'].append(mixedv_best)
        return res


    def reverse(self, model, ds, target, device='cpu'):
        dl = torch.utils.data.DataLoader(ds, 32, shuffle=False, num_workers=0, drop_last=False)
        dl_re = self.get_dataloader_re(ds, target)
        all_features, weight_map_class, init_mask, feature_shape = self.get_feature_info(model, dl, self.layer_name, device)
        mask_tanh = torch.nn.Parameter(init_mask)
        feats_hook = []

        AE = UNet(n_channels=3, num_classes=3, base_filter_num=32, num_blocks=4).to(device)
        AE.train()

        weight_p, weight_acc, weight_std = 1, 1, 1
        optimizerR = torch.optim.Adam(AE.parameters(), lr=0.001, betas=(0.5, 0.9))
        optimizerR_mask = torch.optim.Adam([mask_tanh], lr=1e-1, betas=(0.5, 0.9))
        criterion_ce = torch.nn.CrossEntropyLoss()  # .cuda()
        criterion_mse = torch.nn.MSELoss(reduction='mean')


        def forward_ae(x):
            x_ae = AE(x)
            out = model(x_ae)
            # features = feats_hook
            features = feats_hook.pop()
            return out, features, x, x_ae
        
        def set_hook(model, feats_hook, layer_name, wp=True):
            def hook0(module, feat_in, feat_out):
                # used in epochs_wp
                feats_hook.append(feat_out)
                return None
            
            def hook1(module, feat_in, feat_out):
                feats_hook.append(feat_out)
                mask = self.get_raw_mask(mask_tanh)
                ref_features_idx = np.random.choice(range(all_features.shape[0]), feat_out.shape[0], replace=True)
                ref_features = all_features[ref_features_idx]
                features = mask * feat_out + (1 - mask) * ref_features.reshape(feat_out.shape)
                return features

            hook = hook0 if wp else hook1
            for (name, module) in model.named_modules():
                if name == layer_name: 
                    handle = module.register_forward_hook(hook=hook)
                    return handle
            raise Exception('FeatureRE.reverse.set_hook(): layer[{}] not found.'.format(layer_name))


        mixed_value_best = float("inf")
        # Learning the transformation
        handle = set_hook(model, feats_hook, self.layer_name, wp=True)
        print("FeatureRE: Reversing target {}".format(target)) 
        for epoch in range(self.epochs):
            if epoch == self.epochs_wp:
                # set hook to extract & modify feature
                handle.remove()
                handle = set_hook(model, feats_hook, self.layer_name, wp=False)
                print('FeatureRE: Warm up ended at [{}/{}]'.format(self.epochs_wp, self.epochs))

            total_pred = 0
            true_pred = 0
            loss_ce_list, loss_dist_list, loss_list = [], [], []
            acc_list = []
            loss_p_list, loss_mask_norm_list, loss_std_list = [], [], []
            loss_p_bound, loss_std_bound, asr_thres = 0.15, 1.0, 0.9
            mask_norm_bound = int(reduce(lambda x, y: x * y, feature_shape) * 0.03)  # 15

            for batch_idx, (inputs, labels) in enumerate(dl_re):
                AE.train()
                mask_tanh.requires_grad = False
                optimizerR.zero_grad()
                inputs = inputs.to(device)
                sample_num = inputs.shape[0]
                total_pred += sample_num
                target_labels = torch.ones(sample_num, dtype=torch.int64, device=device) * target

                predictions, features, x_before_ae, x_after_ae = forward_ae(inputs)
                loss_ce = criterion_ce(predictions, target_labels)
                loss_mse = criterion_mse(x_after_ae, x_before_ae)
                loss_dist = torch.cosine_similarity(weight_map_class[target].reshape(-1), features.mean(0).reshape(-1), dim=0)

                batch_acc = torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach() / sample_num
                acc_list.append(batch_acc)
                avg_acc_G = batch_acc
                loss_p = loss_mse
                
                if epoch < self.epochs_wp:
                    total_loss = loss_ce
                    if loss_p > loss_p_bound:
                        total_loss += loss_p * 100
                else:
                    loss_std = (features * self.get_raw_mask(mask_tanh)).std(0).sum()
                    loss_std = loss_std / (torch.norm(self.get_raw_mask(mask_tanh), 1))
                    total_loss = loss_dist * 5
                    if loss_dist < 0:
                        total_loss -= loss_dist * 5
                    if loss_std > loss_std_bound:
                        total_loss += loss_std * 10 * (1 + weight_std)
                    if loss_p > loss_p_bound:
                        total_loss += loss_p * 10 * (1 + weight_p)
                    if avg_acc_G.item() < asr_thres:
                        total_loss += 1 * loss_ce * (1 + weight_acc)

                total_loss.backward()
                optimizerR.step()

                # optimize feature mask
                if epoch >= self.epochs_wp:
                    for k in range(self.epochs_mask):
                        AE.eval()
                        mask_tanh.requires_grad = True
                        optimizerR_mask.zero_grad()
                        predictions, features, x_before_ae, x_after_ae = forward_ae(inputs)
                        loss_mask_ce = criterion_ce(predictions, target_labels)
                        loss_mask_norm = torch.norm(self.get_raw_mask(mask_tanh), 1)
                        loss_mask_total = loss_mask_ce
                        if loss_mask_norm > mask_norm_bound:
                            loss_mask_total += loss_mask_norm
                        loss_mask_total.backward()
                        optimizerR_mask.step()
                else:
                    loss_p, loss_mask_norm, loss_std = torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)

                loss_ce_list.append(loss_ce.detach())
                loss_dist_list.append(loss_dist.detach())
                loss_list.append(total_loss.detach())
                loss_p_list.append(loss_p)
                loss_mask_norm_list.append(loss_mask_norm)
                loss_std_list.append(loss_std)
                true_pred += torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach()

            # calculate avg loss & asr
            avg_loss_ce = torch.stack(loss_ce_list).mean()
            avg_loss_dist = torch.stack(loss_dist_list).mean()
            avg_loss = torch.stack(loss_list).mean()
            avg_acc = torch.stack(acc_list).mean()
            avg_loss_p = torch.stack(loss_p_list).mean()
            avg_loss_mask_norm = torch.stack(loss_mask_norm_list).mean()
            avg_loss_std = torch.stack(loss_std_list).mean()
            asr = true_pred * 100.0 / total_pred

            if epoch >= self.epochs_wp:
                # if avg_acc.item() < asr_thres:
                #     print("@avg_asr lower than bound")
                # if avg_loss_p > 1.0 * loss_p_bound:
                #     print("@avg_loss_p larger than bound")
                # if avg_loss_mask_norm > 1.0 * mask_norm_bound:
                #     print("@avg_loss_mask_norm larger than bound")
                # if avg_loss_std > 1.0 * loss_std_bound:
                #     print("@avg_loss_std larger than bound")

                mixed_value = avg_loss_dist.detach() - avg_acc + \
                    max(avg_loss_p.detach() - loss_p_bound, 0) / loss_p_bound + \
                    max(avg_loss_mask_norm.detach() - mask_norm_bound, 0) / mask_norm_bound + \
                    max(avg_loss_std.detach() - loss_std_bound, 0) / loss_std_bound
                # print("mixed_value:", mixed_value)
                if mixed_value < mixed_value_best:
                    mixed_value_best = mixed_value
                weight_p = max(avg_loss_p.detach() - loss_p_bound, 0) / loss_p_bound
                weight_acc = max(asr_thres - avg_acc, 0) / asr_thres
                weight_std = max(avg_loss_std.detach() - loss_std_bound, 0) / loss_std_bound
            else:
                mixed_value = float("inf")

            print(
                "[{:>3}/{}]: ASR: {:>7.3f} | Loss [CE:{:.6f}  Dist:{:.6f}  P:{:.6f}  Mask:{:.6f}  Std:{:.6f}]| MixedV: [cur:{:.4f}  best:{:.4f}]".format(
                    epoch, self.epochs, asr, 
                    avg_loss_ce, avg_loss_dist, avg_loss_p, avg_loss_mask_norm, avg_loss_std, mixed_value, mixed_value_best
                )
            )
        

        return AE, asr.item(), mixed_value.item(), mixed_value_best.item()
        
        
    def get_dataloader_re(self, ds, target):
        ds_re = copy.copy(ds)
        ds_re.data = list(filter(lambda x: x[1] != target, ds.data))
        dl_re = torch.utils.data.DataLoader(ds_re, batch_size=100, pin_memory=True, shuffle=True)
        return dl_re

    def get_feature_info(self, model, dl, layer_name, device):
        features, labels, _ = get_features2(dl, model, layer_name, device=device, loc='out', flatten=False)

        # categorize feats by class & cal mean
        features_cls = [features[torch.where(labels == i)] for i in range(self.num_classes)]
        weight_map_cls = [features_cls[i].mean(0) for i in range(self.num_classes)]

        # generate mask
        feature_shape = list(features.shape[1:])
        mask = torch.ones(feature_shape).to(device)

        return features, weight_map_cls, mask, feature_shape

    def get_raw_mask(self, mask_tanh):
        mask = torch.nn.Tanh()(mask_tanh)
        bounded = mask / (2 + self._EPSILON) + 0.5
        return bounded


    @staticmethod
    def tanh(x: torch.Tensor) -> torch.Tensor:
        return (x.tanh() + 1) * 0.5

    @staticmethod
    def atan_func(x: torch.Tensor) -> torch.Tensor:
        return x.atan().div(math.pi).add(0.5)

    

    def dump_stats(self, path, info_model=None, model_dir=None):
        import json
        os.makedirs(path, exist_ok=True)
        info = {'config': self.config, 'res': self.res, 'model': info_model, 'model_dir': model_dir}

        json.dump(info, open(os.path.join(path, 'stats.json'), 'w'), cls=CompactJSONEncoder, indent=4)
        print('FeatureRE stats saved to: ' + path)


if __name__ == "__main__":
    pass


