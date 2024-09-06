import os
import random
import pickle

import numpy as np
from PIL import Image
import torch


class Blended:
    def __init__(self, tgt_ls, img_shape, pattern=None, mask_ratio=0.3):
        self.tgt_ls = tgt_ls
        self.img_shape = img_shape
        self.mask_ratio = mask_ratio

        self.pattern = pattern
        self.pattern_img = self.load_pattern(pattern)
        self.config = {'type': 'Blended', 'targets': tgt_ls, 'pattern': pattern, 'mask_ratio': mask_ratio}

    def __call__(self, x, *args, **kwargs):
        img, tgt = x[0], random.choice(self.tgt_ls)
        img = img * (1 - self.mask_ratio) + self.mask_ratio * self.pattern_img
        
        return img, tgt
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[0:2]
        pattern = self.pattern.resize((w, h))
        pattern = np.array(pattern)

        # for RGBA
        if pattern.shape[-1] == 4:
            mr = self.mask_ratio * pattern[:, :, -1:] / 255
            pattern = pattern[:, :, :-1]
        else:
            mr = self.mask_ratio

        image = image * (1 - mr) + mr * pattern
        return image.astype(np.uint8)

    def load_pattern(self, pattern):
        if pattern in [None, 'hellokitty', 'HelloKitty']:
            pattern = 'utils/assets/HelloKitty.png'

        elif pattern in ['noise', 'Noise']:
            pattern = 'utils/assets/noise.jpg'
        elif pattern in ['noise2', 'Noise2']:
            pattern = 'utils/assets/noise2.jpg'
        elif pattern in ['noise2png', 'Noise2png']:
            pattern = 'utils/assets/noise2.png'
        elif pattern in ['noise2blur', 'Noise2Blur']:
            pattern = 'utils/assets/noise2blur.jpg'
        elif pattern in ['noise2blurpng', 'Noise2Blurpng']:
            pattern = 'utils/assets/noise2blur.png'
        
        elif pattern in ['noise3', 'Noise3']:
            pattern = 'utils/assets/noise3.jpg'
        elif pattern in ['noise3png', 'Noise3png']:
            pattern = 'utils/assets/noise3.png'
        elif pattern in ['noise4', 'Noise4']:
            pattern = 'utils/assets/noise4.jpg'
        elif pattern in ['noise4png', 'Noise4png']:
            pattern = 'utils/assets/noise4.png'
        elif pattern.startswith('noise2jpg'):
            pattern = 'utils/assets/' + pattern + '.jpg'

        elif pattern in ['Red', 'red']:
            pattern = 'utils/assets/red.png'
        elif pattern in ['Green', 'green']:
            pattern = 'utils/assets/green.png'
        elif pattern in ['Blue', 'blue']:
            pattern = 'utils/assets/blue.png'
            
        elif pattern in ['ChessBoard', 'chessboard']:
            pattern = 'utils/assets/chessboard0.png'
        elif pattern in ['bars0', 'bars1', 'bars2', 'bars3']:
            pattern = 'utils/assets/' + pattern + '.png'

        else:
            if isinstance(pattern, str):
                raise Exception('Blended: pattern not implemented:', pattern)
            return pattern
        pattern_img = Image.open(pattern).resize(self.img_shape[-2:])
        if self.img_shape[-3] == 1:
            pattern_img = pattern_img.convert('L')
        return np.atleast_3d(np.array(pattern_img) / 255)

    def visualize(self, path, verbose='', show=False):
        os.makedirs(path, exist_ok=True)
        img = (self.mask_ratio * self.pattern_img * 255).squeeze().astype(np.uint8)
        img = Image.fromarray(img)
        img.save(os.path.join(path, 'pattern_{}.png'.format(verbose,)))
        if show:
            img.show()
        print('Blended: pattern saved to {}'.format(path))
    
    @property
    def name(self):
        # blended_r0.1_mr0.7_hellokitty
        return '_'.join(['blended', str(self.pattern), 'mr'+str(self.mask_ratio), ]).strip('_')
