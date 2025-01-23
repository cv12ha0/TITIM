import os
import math
import pickle
import typing

from PIL import Image
import numpy as np
import torch

import wand
import wand.image
import wand.color
import wand.drawing


class Styled:
    """
    filter from:
    https://github.com/ionutmodo/TrojAI-UMD/blob/6ee5912f1fa57f49a4dd4feeeaf7862153bb6a9f/trojai/trojai/datagen/instagram_xforms.py
    https://github.com/naashonomics/steamlit_projects/blob/main/opencv/instagram_filters/instagram_filters.py
    http://qinxuye.me/article/implement-sketch-and-pencil-with-pil/
    """
    def __init__(self, tgt_ls, img_shape, filter_name='gotham', mix_ratio=1.0, framed=False, dtype='float'):
        self.tgt_ls = tgt_ls
        self.img_shape = img_shape
        self.filter_name = filter_name
        self.filter = self.get_filter(filter_name)
        self.mix_ratio = mix_ratio
        self.framed = framed
        self.dtype = dtype
        self.config = {'type': 'Styled', 'targets': tgt_ls, 'filter_name': filter_name, 'mix_ratio': mix_ratio, 'framed': framed}

    def __call__(self, x, *args, **kwargs):
        img, tgt = x[0], self.tgt_ls[0]
        if self.dtype in ['float']:
            img_applied = self.apply_float(img)
            img = self.mix_ratio*img_applied + (1 - self.mix_ratio)*img
        else:
            img_applied = self.apply(img)
            img = (self.mix_ratio*img_applied + (1 - self.mix_ratio)*img).astype(np.uint8)
        return img, tgt

    def get_filter(self, filter_name=None) -> typing.Callable:
        filter_name = filter_name if filter_name else self.filter_name
        if filter_name in ['None', 'none', 'No', 'no']:
            return self.no_filter
        elif filter_name in ['Gotham', 'gotham']:
            return self.gotham_filter
        elif filter_name in ['Nashville', 'nashville']:
            return self.nashville_filter
        elif filter_name in ['Kelvin', 'kelvin']:
            return self.kelvin_filter
        elif filter_name in ['Lomo', 'lomo']:
            return self.lomo_filter
        elif filter_name in ['Toaster', 'toaster']:
            return self.toaster_filter
        else:
            raise Exception('filter not defined: ', filter_name)

    def apply(self, image: np.ndarray) -> np.ndarray:
        image = wand.image.Image.from_array(image)  # np.uint8(image * 255)
        image = self.filter(image)
        # image.background_color = wand.color.Color("white")
        # image.alpha_channel = 'remove'
        image = np.array(image)  # / 255
        return image

    def apply_float(self, image: np.ndarray) -> np.ndarray:
        image = wand.image.Image.from_array(np.uint8(image * 255))
        image = self.filter(image)
        image.alpha_channel = 'remove'
        image = np.array(image)[..., :3] / 255
        return image

    def _colortone(self, image: wand.image.Image, color, dst_percent, invert):
        mask_src = image.clone()
        mask_src.colorspace = 'gray'
        if invert:
            mask_src.negate()
        mask_src.alpha_channel = 'copy'

        src = image.clone()
        src.colorize(wand.color.Color(color), wand.color.Color('#FFFFFF'))

        src.composite_channel('alpha', mask_src, 'copy_alpha')  # 'copy_opacity' for ImageMagick-6
        image.composite_channel('default_channels', src, 'blend', arguments=str(dst_percent) + "," + str(100 - dst_percent))

    def _vignette(self, image: wand.image.Image, color_1: str = 'none', color_2: str = 'black', crop_factor: float = 1.5) -> None:
        # darken corners
        crop_x = math.floor(image.width * crop_factor)
        crop_y = math.floor(image.height * crop_factor)
        src = wand.image.Image()
        src.pseudo(width=crop_x, height=crop_y, pseudo='radial-gradient:' + color_1 + '-' + color_2)
        src.crop(0, 0, width=image.width, height=image.height, gravity='center')
        src.reset_coords()
        image.composite_channel('default_channels', src, 'multiply')
        image.merge_layers('flatten')

    def gotham_filter(self, image: wand.image.Image) -> wand.image.Image:
        filtered_image = image.clone()
        filtered_image.modulate(120, 10, 100)
        filtered_image.colorize(wand.color.Color('#222b6d'), wand.color.Color('#333333'))
        filtered_image.gamma(.9)
        filtered_image.sigmoidal_contrast(True, 3, .5 * filtered_image.quantum_range)
        filtered_image.sigmoidal_contrast(True, 3, .5 * filtered_image.quantum_range)
        if self.framed:
            filtered_image.border(wand.color.Color('#000'), 20, 20)
        return filtered_image

    def nashville_filter(self, image: wand.image.Image) -> wand.image.Image:
        filtered_image = image.clone()
        self._colortone(filtered_image, '#222b6d', 50, True)
        self._colortone(filtered_image, '#f7daae', 50, False)
        filtered_image.sigmoidal_contrast(True, 3, .5 * filtered_image.quantum_range)
        filtered_image.modulate(100, 150, 100)
        filtered_image.auto_gamma()
        if self.framed:
            frame = wand.image.Image(filename='utils/assets/Nashville.jpg')
            frame.resize(filtered_image.width, filtered_image.height)
            filtered_image.sequence.append(frame)
            filtered_image.merge_layers('merge')
        return filtered_image

    def kelvin_filter(self, image: wand.image.Image) -> wand.image.Image:
        filtered_image = image.clone()
        filtered_image.auto_gamma()
        filtered_image.modulate(120, 50, 100)
        with wand.drawing.Drawing() as draw:
            draw.fill_color = '#FF9900'
            draw.fill_opacity = 0.2
            draw.rectangle(left=0, top=0, width=filtered_image.width, height=filtered_image.height)
            draw(filtered_image)
        if self.framed:
            frame = wand.image.Image(filename='utils/assets/Kelvin.jpg')
            frame.resize(filtered_image.width, filtered_image.height)
            filtered_image.sequence.append(frame)
            filtered_image.merge_layers('merge')
        return filtered_image

    def lomo_filter(self, image: wand.image.Image) -> wand.image.Image:
        filtered_image = image.clone()
        filtered_image.level(.5, channel="R")
        filtered_image.level(.5, channel="G")
        self._vignette(filtered_image)
        return filtered_image

    def toaster_filter(self, image: wand.image.Image) -> wand.image.Image:
        filtered_image = image.clone()
        self._colortone(filtered_image, '#330000', 50, True)
        filtered_image.modulate(150, 80, 100)
        filtered_image.gamma(1.2)
        filtered_image.sigmoidal_contrast(True, 3, .5 * filtered_image.quantum_range)
        filtered_image.sigmoidal_contrast(True, 3, .5 * filtered_image.quantum_range)
        self._vignette(filtered_image, 'none', 'LavenderBlush3')
        self._vignette(filtered_image, '#ff9966', 'none')
        return filtered_image

    def no_filter(self, image: wand.image.Image) -> wand.image.Image:
        return image

    def sketch(self, img: Image, threshold=15) -> Image:
        threshold = min(100, max(0, threshold))
        width, height = img.size
        img = img.convert('L')  # convert to grayscale mode
        pix = img.load()  # get pixel matrix

        for w in range(width):
            for h in range(height):
                if w == width - 1 or h == height - 1:
                    continue

                src = pix[w, h]
                dst = pix[w + 1, h + 1]

                diff = abs(src - dst)

                if diff >= threshold:
                    pix[w, h] = 0
                else:
                    pix[w, h] = 255
        return img

    def pencil(self, img: Image, threshold=15) -> Image:
        threshold = min(100, max(0, threshold))
        width, height = img.size
        dst_img = Image.new("RGBA", (width, height))

        if img.mode != "RGBA":
            img = img.convert("RGBA")

        pix = img.load()
        dst_pix = dst_img.load()

        for w in range(width):
            for h in range(height):
                if w == 0 or w == width - 1 or h == 0 or h == height - 1:
                    continue
                # include 9 pixels around cur pixel
                around_wh_pixels = [pix[i, j][:3] for j in range(h - 1, h + 2) for i in range(w - 1, w + 2)]
                # exclude current pixel
                exclude_wh_pixels = tuple(around_wh_pixels[:4] + around_wh_pixels[5:])
                # take avg of pixels
                RGB = list(map(lambda l: int(sum(l) / len(l)), zip(*exclude_wh_pixels)))

                cr_p = pix[w, h]  # current pixel
                cr_draw = all([abs(cr_p[i] - RGB[i]) >= threshold for i in range(3)])

                if cr_draw:
                    dst_pix[w, h] = 0, 0, 0, cr_p[3]
                else:
                    dst_pix[w, h] = 255, 255, 255, cr_p[3]
        return dst_img

    def visualize_one(self, tgt, idx, path, verbose='', show=False):
        # img = (mask * pattern * 255).squeeze().astype(np.uint8)
        # img = Image.fromarray(img)
        # img.save(os.path.join(path, '{}{}_{}.png'.format(verbose, tgt, idx)))
        # if show:
        #     img.show()
        pass

    def visualize(self, path, verbose=''):
        # os.makedirs(path, exist_ok=True)
        # for tgt in self.tgt_ls:
        #     for idx in range(self.pattern_per_target):
        #         self.visualize_one(tgt, idx, path, verbose)
        # print('Styled: all patterns saved to {}'.format(path))
        pass

    @property
    def name(self):
        # styled_r0.1_gotham_wof
        framed = 'wf' if self.framed is True else 'wof'
        return '_'.join(['styled', self.filter_name, 'mr'+str(self.mix_ratio), framed, ]).strip('_')
