import os
import random

import numpy as np
from PIL import Image


class Square:
    def __init__(self, tgt_ls, img_shape, block_size=4, margin=1, mask_ratio=1, color=None, inject_ratio=0):
        self.tgt_ls = tgt_ls
        self.img_shape = img_shape
        self.block_size = block_size
        self.margin = margin
        self.mask_ratio = mask_ratio
        self.color = color
        self.inject_ratio = inject_ratio
        self.config = {'type': 'Square', 'targets': tgt_ls, 'block_size': block_size, 'margin': margin,
                       'mask_ratio': mask_ratio, 'color': color}

        # construct pattern
        self.pattern_dict = {}
        for y_target in self.tgt_ls:
            cur_ls = []
            mask, pattern = self.construct_block(block_size, margin, color)
            cur_ls.append([mask, pattern])
            self.pattern_dict[y_target] = cur_ls

    def __call__(self, x, *args, **kwargs):
        # if random.uniform(0, 1) > self.inject_ratio:
        #     return x
        img, tgt = x[0], random.choice(self.tgt_ls)
        mask, pattern = random.choice(self.pattern_dict[tgt])
        mask, pattern = mask * self.mask_ratio, pattern
        img = img * (1 - mask) + mask * pattern
        return img, tgt

    def construct_block(self, block_size=4, margin=1, color=None):
        channel, row, col = self.img_shape
        color = [1]*channel  # TODO: if color is None else color
        mask, pattern = np.zeros((row, col, 1)), np.zeros((row, col, channel))

        mask[row-margin-block_size:row-margin, col-margin-block_size:col-margin, :] = 1  # self.mask_ratio
        for i in range(channel):
            pattern[:, :, i] = color[i]
        # pattern = np.clip(mask * pattern, 0, 1)
        pattern = np.clip(pattern, 0, 1)

        return mask, pattern

    def visualize(self, path, verbose=''):
        os.makedirs(path, exist_ok=True)
        for tgt in self.tgt_ls:
            for idx, (mask, pattern) in enumerate(self.pattern_dict[tgt]):
                img = (mask * pattern * 255).squeeze().astype(np.uint8)
                img = Image.fromarray(img)
                img.save(os.path.join(path, '{}{}_{}.png'.format(verbose, tgt, idx)))
    
    @property
    def name(self):
        # square_r0.1_b4_m1_mr1
        # TODO: add color
        return '_'.join(['square', 'b'+str(self.block_size), 'm'+str(self.margin), 'mr'+str(self.mask_ratio), ]).strip('_')
