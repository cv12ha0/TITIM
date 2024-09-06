import os
import sys
import random
import math

import numpy as np
from PIL import Image

import torch


class BadNets:
    def __init__(self, tgt_ls, img_shape, block_size=4, block_num=3, mask_ratio=1, pattern_per_target=1, fixed=False):
        self.tgt_ls = tgt_ls
        self.img_shape = img_shape
        self.block_size = block_size
        self.block_num = block_num
        self.mask_ratio = mask_ratio
        self.pattern_per_target = pattern_per_target
        self.fixed = fixed
        self.config = {'type': 'BadNets', 'targets': tgt_ls, 'block_size': block_size, 'block_num': block_num,
                       'pattern_per_target': pattern_per_target, 'mask_ratio': mask_ratio, 'fixed': fixed}

        # load / construct pattern
        self.pattern_dict, load_idx = {}, 0
        fname = '_'.join(['badnets', 'b'+str(block_size), 'bn'+str(block_num), 'sz'+str(img_shape[1])])
        for y_target in self.tgt_ls:
            cur_ls = []
            for idx in range(pattern_per_target):
                fname_cur = 'utils/assets/' + fname + '_' + str(load_idx) + '.png'
                if fixed and os.path.exists(fname_cur):
                    img = np.array(Image.open(fname_cur)) / 255  # .resize(img_shape[1:])
                    pattern, mask = img[:, :, :self.img_shape[0]], img[:, :, self.img_shape[0]:]   # * self.mask_ratio
                    load_idx += 1
                    print('Badnets:  fname', fname)
                else:
                    mask = np.zeros((self.img_shape[1], self.img_shape[2], 1))
                    for _ in range(block_num):
                        mask += self.construct_block_mask(block_size)
                    mask = np.clip(mask, 0, 1)

                    mean, std = 0.5, 0.5  # random.uniform(0, 1), random.uniform(0, 1)
                    pattern = np.random.normal(mean, std, (self.img_shape[1], self.img_shape[2], self.img_shape[0]))
                    pattern = np.clip(pattern * mask, 0, 1)
                    if fixed:
                        img = (np.dstack([pattern, (mask != 0).astype(float)])*255).astype(np.uint8)
                        Image.fromarray(img).save(fname_cur)  # save pattern
                cur_ls.append([mask, pattern])
                # save
                self.pattern_dict[y_target] = cur_ls


    def __call__(self, x, *args, **kwargs):
        img, tgt = x[0], random.choice(self.tgt_ls)
        mask, pattern = random.choice(self.pattern_dict[tgt])
        mask, pattern = mask * self.mask_ratio, pattern
        img = img * (1 - mask) + mask * pattern
        return img, tgt

    def construct_block_mask(self, block_size=3):
        channel, row, col = self.img_shape
        b_col = random.choice(range(0, col - block_size + 1))
        b_row = random.choice(range(0, row - block_size + 1))

        mask = np.zeros((row, col, 1))
        mask[b_row:b_row + block_size, b_col:b_col + block_size, :] = 1  # self.mask_ratio
        return mask

    def visualize_one(self, tgt, idx, path, verbose='', show=False):
        mask, pattern = self.pattern_dict[tgt][idx]
        img = (mask * pattern * 255).squeeze().astype(np.uint8)
        img = Image.fromarray(img)
        img.save(os.path.join(path, '{}{}_{}.png'.format(verbose, tgt, idx)))
        if show:
            img.show()

    def visualize(self, path, verbose=''):
        os.makedirs(path, exist_ok=True)
        for tgt in self.tgt_ls:
            for idx in range(self.pattern_per_target):
                self.visualize_one(tgt, idx, path, verbose)
        print('BadNets: all patterns saved to {}'.format(path))
    
    @property
    def name(self):
        # badnets_r0.1_b4_bn3_mr1_ppt1
        return '_'.join(['badnets', 'b'+str(self.block_size), 'bn'+str(self.block_num), 'ppt'+str(self.pattern_per_target), 'mr'+str(self.mask_ratio), ]).strip('_')
