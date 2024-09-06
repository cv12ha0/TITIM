import os
import sys
import random

import numpy as np
from PIL import Image


class SIG:
    def __init__(self, tgt_ls, img_shape, delta=40, f=6):
        self.tgt_ls = tgt_ls
        self.img_shape = img_shape
        self.delta = delta
        self.f = f

        self.pattern = self.load_pattern(delta, f)
        self.config = {'type': 'SIG', 'targets': tgt_ls, 'delta': delta, 'f': f}


    def __call__(self, x, *args, **kwargs):
        img, tgt = x[0], random.choice(self.tgt_ls)
        img = np.clip(img + self.pattern, 0, 1)
        return img, tgt
    
    def load_pattern(self, delta=40, f=60):
        pattern = np.zeros((self.img_shape[1], self.img_shape[2]))
        m = self.img_shape[2]
        for i in range(self.img_shape[1]):
            for j in range(self.img_shape[2]):
                pattern[i, j] = self.delta * np.sin(2 * np.pi * j * self.f / m)

        pattern = (pattern / 255).astype(np.float32)[:, :, np.newaxis]
        return pattern
    

    def visualize(self, path, verbose='', show=False):
        os.makedirs(path, exist_ok=True)
        img = (np.clip(self.pattern, 0, 1) * 255).squeeze().astype(np.uint8)
        img = Image.fromarray(img)
        img.save(os.path.join(path, 'pattern_{}.png'.format(verbose, )))
        if show:
            img.show()
        print('SIG: pattern saved to {}'.format(path))
    
    @property
    def name(self):
        # sig_d40_f6
        return '_'.join(['sig', 'd'+str(self.delta), 'f'+str(self.f), ]).strip('_')
