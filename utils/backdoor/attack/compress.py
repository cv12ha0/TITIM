import os
import sys
import random
import math
import io

import numpy as np
from PIL import Image
import cv2

import torch


class Compress:
    def __init__(self, tgt_ls, img_shape, alg='jpeg', quality=100):
        self.tgt_ls = tgt_ls
        self.img_shape = img_shape
        self.alg = alg
        self.quality = quality
        self.compress = self.get_compression(alg)

        self.config = {'type': 'BadNets', 'targets': tgt_ls, 'alg': alg, 'quality': quality}


    def __call__(self, x, *args, **kwargs):
        img, tgt = x[0], random.choice(self.tgt_ls)
        img = self.compress(img.squeeze(), self.quality)
        if img.shape[-1] == 3 and self.img_shape[0] == 1:
            img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        img = np.atleast_3d(img)  # for mnist

        return img, tgt


    def get_compression(self, alg):
        if alg in ['none', 'png']:
            return lambda x: x
        elif alg in ['pil', 'jpgpil', 'jpegpil']:
            return self.compression_pil
        elif alg in ['cv', 'jpgcv', 'jprgcv']:
            return self.compression_cv
            
        pass


    @staticmethod
    def compression_pil(image: Image, quality) -> Image:
        image = Image.fromarray((image*255).astype(np.uint8))
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=quality)  # optimize=True
        image = np.array(Image.open(buffered)).astype(np.float32) / 255
        return image

    @staticmethod
    def compression_cv(image: Image, quality) -> Image:
        image = (image*255).astype(np.uint8)
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        msg = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
        img_compressed = cv2.imdecode(msg, cv2.IMREAD_COLOR)

        img_rgb = cv2.cvtColor(img_compressed, cv2.COLOR_RGB2BGR)
        image = img_rgb.astype(np.float32) / 255
        return image


    def visualize(self, path, verbose=''):
        print('Compress: visualization not implemented')
    
    @property
    def name(self):
        return '_'.join(['compress', self.alg, 'q'+str(self.quality)]).strip('_')
