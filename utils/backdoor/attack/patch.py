import os
import random
import pickle

import numpy as np
from PIL import Image
import torch


class Patch:
    def __init__(self, tgt_ls, img_shape, pattern=None, mask_ratio=1., size=4, loc=28):
        self.tgt_ls = tgt_ls
        self.img_shape = img_shape
        self.mask_ratio = mask_ratio
        self.size = size if isinstance(size, (list, tuple)) else [size, size]
        self.loc = loc if isinstance(loc, (list, tuple)) else [loc, loc]

        self.pattern_name = pattern
        self.pattern, self.mask = self.load_pattern(pattern)
        self.mask *= self.mask_ratio
        self.config = {'type': 'Patch', 'targets': tgt_ls, 'pattern': pattern, 'mask_ratio': mask_ratio, 'size': self.size, 'loc': self.loc}

    def __call__(self, x, *args, **kwargs):
        img, tgt = x[0], random.choice(self.tgt_ls)
        img = img * (1 - self.mask) + self.mask * self.pattern

        return img, tgt


    def load_pattern(self, pattern):
        if pattern in [None, 'bomb', 'Bomb']:
            pattern = 'utils/assets/bomb.png'
        elif pattern in ['flower', 'Flower']:
            pattern = 'utils/assets/flower.png'
        elif pattern in ['pokemon', 'Pokemon']:
            pattern = 'utils/assets/pokemon.png'
        elif pattern in ['flowerhalf', 'FlowerHalf']:
            pattern = 'utils/assets/flowerhalf.png'
        elif pattern in ['flowerfan', 'FlowerFan']:
            pattern = 'utils/assets/flowerfan.png'

        elif pattern in ['ringw', 'RingW', 'RingWhite']:
            pattern = 'utils/assets/ring_white.png'
        elif pattern in ['ringb', 'RingB', 'RingBlack']:
            pattern = 'utils/assets/ring_black.png'
        elif pattern in ['ringws', 'RingWS', 'RingWhiteSolid']:
            pattern = 'utils/assets/ring_white_solid.png'
        elif pattern in ['ringbs', 'RingBS', 'RingBlackSolid']:
            pattern = 'utils/assets/ring_black_solid.png'

        else:
            if isinstance(pattern, str):
                raise Exception('Patch: pattern not found:', pattern)
            return pattern
        
        c, h, w = self.img_shape
        pattern_img = Image.open(pattern).resize(self.size)
        pattern_img = np.array(pattern_img) / 255
        x1, x2, y1, y2 = 0-self.loc[0], h-self.loc[0], 0-self.loc[1], w-self.loc[1]
        pattern_img = pattern_img[x1:x2, y1:y2]  # crop pattern that exceeds

        # pattern, mask = pattern[..., :3], pattern[..., 3]
        pattern = np.zeros((h, w, 3))
        mask = np.zeros((h, w, 1))
        pattern[self.loc[0]:self.loc[0]+self.size[0], self.loc[1]:self.loc[1]+self.size[1], :3] = pattern_img[..., :3]
        mask[self.loc[0]:self.loc[0]+self.size[0], self.loc[1]:self.loc[1]+self.size[1], :] = pattern_img[..., 3:]

        if c == 1: 
            pattern = np.atleast_3d(0.299 * pattern[..., 0] + 0.587 * pattern[..., 1] + 0.114 * pattern[..., 2])
        return pattern, mask

    def visualize(self, path, verbose='', show=False):
        os.makedirs(path, exist_ok=True)
        img = (self.mask_ratio * self.pattern_img * 255).squeeze().astype(np.uint8)
        img = Image.fromarray(img)
        img.save(os.path.join(path, 'pattern_{}.png'.format(verbose,)))
        if show:
            img.show()
        print('Patch: pattern saved to {}'.format(path))
    
    @property
    def name(self):
        # patch_bomb_mr1.0_mr0.7_sz2_loc30_0.1
        return '_'.join(['patch', str(self.pattern_name), 'mr'+str(self.mask_ratio), 'sz'+str(self.size[0]), 'loc'+str(self.loc[0])]).strip('_')
    

