'''
    Anti-Backdoor Learning: Training Clean Models on Poisoned Data (https://proceedings.neurips.cc/paper/2021/hash/7d38b1e9bd793d3f45e0e212a729a93c-Abstract.html)

    code:
        Official (https://github.com/bboylyg/ABL)
        BackdoorBox (https://github.com/THUYimingLi/BackdoorBox/blob/main/core/defenses/ABL.py)
'''
import os
import sys
import warnings
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from ...misc import get_optimizer, get_scheduler, fit

__all__ = [
    'ABL', 
]


class ABL:
    def __init__(self, img_shape, num_classes, tgt_ls=None, gamma=0.5, device='cpu'):
        self.img_shape = img_shape
        self.c, self.h, self.w = img_shape
        self.num_classes = num_classes
        self.gamma = gamma
        self.device = device

        self.tgt_ls = tgt_ls if tgt_ls else list(range(num_classes))

        self.config = {'type': 'Anti-BackdoorLearning', 'img_shape': img_shape, 'num_classes': num_classes, 'tgt_ls': tgt_ls, 
                       'gamma': gamma, }

    def train(self, ds_train, ds_vals, model, epochs, lr, criterion, optimizer, scheduler, batch_size, num_workers, output_dir):
        criterion_isolation = LGALoss(criterion, self.gamma)
        optimizer = get_optimizer(optimizer, [p for p in model.parameters() if p.requires_grad], lr)
        scheduler = get_scheduler(scheduler, optimizer, epochs, milestones=[])

        dl_train = DataLoader(ds_train, batch_size, shuffle=True, num_workers=num_workers, drop_last=False, pin_memory=True)
        dl_vals = [DataLoader(ds_val, batch_size, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=True) for ds_val in ds_vals]

        model = fit(model, epochs, criterion_isolation, optimizer, dl_train, dl_vals, scheduler, self.device, batch_size, output_dir=output_dir, plot=True, disable_prog=True)
        return model


    def filter(self, ds, model, criterion, split_ratio, batch_size=32, num_workers=0, poison_rate=None):
        """Split dataset into poisoned & clean part"""
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        losses = []
        with torch.no_grad():
            for batch_x, batch_y in dl:
                output = model(batch_x.to(self.device))
                loss = criterion(output, batch_y.to(self.device))
                losses.append(loss)
        losses = torch.cat(losses, dim=0)
        num_filter = int(split_ratio * len(losses))
        indices = torch.argsort(losses)
        suspicious_indices = indices[:num_filter]

        # TODO: succ_rate calculation
        if poison_rate is not None:
            num_poison = int(len(ds) * poison_rate)
            num_correct = torch.sum(suspicious_indices < num_poison).item()
            num_recall = torch.sum(indices[:num_poison] < num_poison).item()
            succ_rate = num_correct / num_filter
            succ_rate2 = num_recall / num_poison  # recall

            print('abl: succ:{:.2f} [{}/{}]    succ2:{:.2f} [{}/{}]'
                  .format(succ_rate, num_correct, num_filter, succ_rate2, num_recall, num_poison))
        self.suspicious_indices = suspicious_indices
        return suspicious_indices, [round(succ_rate, 4), num_correct, num_filter, round(succ_rate2, 4), num_recall, num_poison]


    def unlearn():
        # TODO: loss *= -1 for unlearning.
        pass


    def dump_stats(self, path, info_model=None, model_dir=None):
        import json
        os.makedirs(path, exist_ok=True)
        info = {'config': self.config, 'res': self.res, 'model': info_model, 'model_dir': model_dir}

        json.dump(info, open(os.path.join(path, 'stats.json'), 'w'), indent=4)
        print('Anti-Backdoor Learning stats saved to: ' + path)


class LGALoss(torch.nn.Module):
    def __init__(self, loss, gamma):
        """The local gradient ascent (LGA) loss used in first phrase (called pre-isolation phrase) in ABL.

        Args:
            loss (nn.Module): Loss for repaired model training. Please specify the reduction augment in the loss.
            gamma (float): Lower Bound for repairing model    
        """
        super().__init__()
        self.loss = loss
        self.gamma = gamma
        
        if not hasattr(loss, 'reduction'):
            raise ValueError("Loss module must have the attribute named reduction!")
        if loss.reduction not in ['none', 'mean']:
            raise NotImplementedError("This loss only support loss.reduction='mean' or loss.reduction='none'")
    
    def forward(self, logits, targets):
        loss = self.loss(logits, targets)
        if self.loss.reduction == 'none':
            loss = loss.mean()    
        loss = torch.sign(loss-self.gamma) * loss  
        return loss



if __name__ == "__main__":
    # activation_clustering = ActivationClustering(args.img_shape, args.num_classes, device=device)
    # activation_clustering.clustering(ds, model, layer_name='avgpool')  # layer4  conv_block3
    # activation_clustering.dump_stats(os.path.join('data', args.dataset, args.subset, 'ActivationClustering'), model.config, model_dir)
    pass


