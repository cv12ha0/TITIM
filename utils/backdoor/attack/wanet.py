import os
import random
import pickle

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F


class WaNet:
    def __init__(self, tgt_ls, img_shape, cross_ratio=2, s=0.5, k=4, grid_rescale=1, fixed=True):
        self.tgt_ls = tgt_ls
        self.img_shape = img_shape
        self.cross_ratio = cross_ratio  # rho_a = inject_ratio, rho_n = inject_ratio * cross_ratio
        self.rho_a = 1 / (1 + cross_ratio)
        self.s = s
        self.k = k
        self.grid_rescale = grid_rescale
        self.fixed = fixed
        self.config = {'type': 'WaNet', 'targets': tgt_ls, 'rho_a': self.rho_a, 'cross_ratio': cross_ratio,
                       's': s, 'k': k, 'grid_rescale': grid_rescale, 'fixed': fixed}
        self.counter_a, self.counter_n = 0, 0

        # prepare grid
        self.grid_a = self.init_grid(s, k, grid_rescale)

    def init_grid(self, s=0.5, k=4, grid_rescale=1, load_idx=0):
        # generate noise grid
        if self.fixed:
            fname = '_'.join(['wanet', 'k'+str(self.k), 'sz'+str(self.img_shape[1])])
            fname = 'utils/assets/' + fname + '_' + str(load_idx) + '.pth'
            if os.path.exists(fname):
                ins = torch.load(fname)
                print("WaNet: loading fixed pattern from ", fname)
            else:
                ins = torch.rand(1, 2, k, k) * 2 - 1
                torch.save(ins, fname)
        else:
            ins = torch.rand(1, 2, k, k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        noise_grid = F.upsample(ins, size=self.img_shape[1], mode="bicubic", align_corners=True).permute(0, 2, 3, 1)  # TODO: nn.functional.interpolate
        # generate identity grid
        array1d = torch.linspace(-1, 1, steps=self.img_shape[1])
        x, y = torch.meshgrid(array1d, array1d)  # TODO: in an upcoming release, it will be required to pass the indexing argument
        identity_grid = torch.stack((y, x), 2)[None, ...]

        # gen grid applied
        grid_a = (identity_grid + s * noise_grid / self.img_shape[1]) * grid_rescale
        grid_a = torch.clamp(grid_a, -1, 1)
        return grid_a

    def __call__(self, x, *args, **kwargs):
        return self.apply_a(x)


    def apply_a(self, x):
        img = torch.tensor(x[0]).permute(2, 0, 1).float()
        img = F.grid_sample(img[None, ...], self.grid_a, align_corners=True)
        tgt = random.choice(self.tgt_ls)

        img = img.squeeze(0).permute(1, 2, 0).numpy()
        return img, tgt

    def apply_n(self, x):
        img = torch.tensor(x[0]).permute(2, 0, 1).float()

        ins = torch.rand(1, self.img_shape[1], self.img_shape[1], 2) * 2 - 1
        grid_n = self.grid_a + ins / self.img_shape[1]
        grid_n = torch.clamp(grid_n, -1, 1)
        img = F.grid_sample(img[None, ...], grid_n, align_corners=True)
        tgt = x[1]

        img = img.squeeze(0).permute(1, 2, 0).numpy()
        return img, tgt
    
    def apply_batch(self, batch_x, batch_y, inject_ratio, cross_ratio, device='cpu'):
        batch_size = batch_x.shape[0]
        poison_num = int(inject_ratio * batch_size)
        cross_num = int(cross_ratio * batch_size)
        grid_a = self.grid_a.to(device)

        # poison
        batch_x[:poison_num, ...] = F.grid_sample(batch_x[:poison_num, ...], grid_a.repeat(poison_num, 1, 1, 1), align_corners=True)
        batch_y[:poison_num, ...] = torch.ones_like(batch_y[:poison_num]) * self.tgt_ls[0]  # random.choice(self.tgt_ls)

        # cross
        ins = (torch.rand(cross_num, self.img_shape[1], self.img_shape[1], 2).to(device) * 2 - 1)
        grid_n = grid_a.repeat(cross_num, 1, 1, 1) + ins / self.img_shape[1]
        grid_n = torch.clamp(grid_n, -1, 1)
        batch_x[poison_num:(poison_num+cross_num), ...] = F.grid_sample(batch_x[poison_num:(poison_num+cross_num), ...], grid_n, align_corners=True)

        return batch_x, batch_y


    def visualize(self, path, verbose=''):
        # print('WaNet: all patterns saved to {}'.format(path))
        print('WaNet: visualize todo.')
        pass
    
    @property
    def name(self):
        # wanet_cr0_s0.5_k4_gs1
        return '_'.join(['wanet', 'cr'+str(self.cross_ratio), 's'+str(self.s), 'k'+str(self.k), 'gs'+str(self.grid_rescale)]).strip('_')
