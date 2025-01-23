"""
    BppAttack (bit-per-pixel)

    BppAttack: Stealthy and Efficient Trojan Attacks Against Deep Neural Networks via Image Quantization and Contrastive Adversarial Learning (https://openaccess.thecvf.com/content/CVPR2022/html/Wang_BppAttack_Stealthy_and_Efficient_Trojan_Attacks_Against_Deep_Neural_Networks_CVPR_2022_paper.html)

    code:
        Official (https://github.com/RU-System-Software-and-Security/BppAttack)
"""

import os
import random
import pickle

import numpy as np
from PIL import Image
import torch


class Bpp:
    depth_original = 8

    def __init__(self, tgt_ls, img_shape, depth=8, mask_ratio=1.0, dither=False):
        self.tgt_ls = tgt_ls
        self.img_shape = img_shape
        self.mask_ratio = mask_ratio

        self.depth = depth
        self.dither = dither
        self.config = {'type': 'Bpp', 'targets': tgt_ls, 'depth': depth, 'mask_ratio': mask_ratio, 'dither': dither}

    def __call__(self, x, *args, **kwargs):
        img, tgt = x[0], random.choice(self.tgt_ls)
        if self.dither:
            img = self.floyd_dither(img, self.depth)
        else:
            img = np.round(img * (2**self.depth)) / (2**self.depth)
            # img = np.round(img/255.*self.depth) / self.depth*255.
        return img, tgt

    # @jit(nopython=True)
    def floyd_dither(self, image, depth):
        channel, h, w = image.shape
        image_out = image.copy()
        for y in range(h):
            for x in range(w):
                old = image_out[:, y, x]
                new = np.round(old * (2**depth)) / (2**depth)
                # temp = np.empty_like(old).astype(np.float64)
                # new = np.round_(old*depth, 0, temp) / depth
                error = old - new
                image_out[:, y, x] = new
                if x + 1 < w:
                    image_out[:, y, x + 1] += error * 0.4375
                if (y + 1 < h) and (x + 1 < w):
                    image_out[:, y + 1, x + 1] += error * 0.0625
                if y + 1 < h:
                    image_out[:, y + 1, x] += error * 0.3125
                if (x - 1 >= 0) and (y + 1 < h): 
                    image_out[:, y + 1, x - 1] += error * 0.1875
        return image_out

    def visualize(self, path, verbose='', show=False):
        # os.makedirs(path, exist_ok=True)
        # img = (self.mask_ratio * self.color_rgb * 255).squeeze().astype(np.uint8)  # single pixel
        # img = img[np.newaxis, np.newaxis, :]  # expand_dim
        # img = img.repeat(self.img_shape[1], 0).repeat(self.img_shape[2], 1)  # repeat to full image

        # img = Image.fromarray(img)
        # img.save(os.path.join(path, 'pattern_{}.png'.format(verbose,)))
        # if show:
        #     img.show()
        # print('Bpp: pattern saved to {}'.format(path))
        pass
    
    @property
    def name(self):
        # bpp_d7_dtT_0.1
        dither = 'T' if self.dither else 'F'
        return '_'.join(['bpp', 'd'+str(self.depth), 'dit'+dither]).strip('_')
