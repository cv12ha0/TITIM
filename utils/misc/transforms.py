import random
import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
import torchvision.transforms.functional as F


class OfficPreProcess:
    def __init__(self, dataset=None, ):
        self.dataset = dataset

    def __call__(self, x, *args, **kwargs):
        image, label = x
        image = np.array(image)
        image = np.atleast_3d(image)
        # if self.dataset in ['mnist', 'MNIST']:
        #     image = image / 255
        image = image / 255

        # image = image.transpose(1, 2, 0)
        # image = image[:, :, ::-1]

        return image, label


class ToNdarray:
    def __init__(self, div255=True):
        self.div255 = div255

    def __call__(self, x, *args, **kwargs):
        image, label = x
        # to ndarray
        image = np.array(image)
        image = np.atleast_3d(image)
        # scale to [0, 1]
        if self.div255:
            image = image / 255

        return image, label


class ToPIL:
    def __init__(self):
        pass

    def __call__(self, x, *args, **kwargs):
        image, label = x
        if isinstance(image, np.ndarray):
            # image = image.transpose((1, 2, 0))
            if image.dtype in [float, np.float16, np.float32, np.float64]:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        return image, label


class Resize:
    def __init__(self, size=224, alg='cv', inter='cubic'):
        self.size = size
        self.alg = alg
        self.inter = self.get_inter(alg, inter)

    def __call__(self, x, *args, **kwargs):
        image, label = x
        image = self._rsz(image)
        return image, label
    
    def get_inter(self, alg, inter):
        if alg.lower() in ['cv', 'opencv']:
            if inter in ['area']:
                return cv2.INTER_AREA
            elif inter in ['cubic']:
                return cv2.INTER_CUBIC
            elif inter in ['linear']:
                return cv2.INTER_LINEAR
            elif inter in ['nearst']:
                return cv2.INTER_NEAREST
        elif alg.lower() in ['pil']:
            if inter in ['nearst']:
                return Image.NEAREST
            elif inter in ['box']:
                return Image.BOX
            elif inter in ['bicubic']:
                return Image.BICUBIC
            elif inter in ['lanczos']:
                return Image.LANCZOS
            
    def _rsz(self, img):
        if self.alg in ['cv', 'opencv']:
            return self._rsz_cv(img)
        elif self.alg in ['pil']:
            return self._rsz_pil(img)

    def _rsz_cv(self, img):
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_rsz = cv2.resize(img_bgr, (self.size, self.size), interpolation=self.inter)
        img_rsz = cv2.cvtColor(img_rsz, cv2.COLOR_BGR2RGB)
        img_rsz = np.clip(img_rsz, 0, 1)
        return img_rsz

    def _rsz_pil(self, img):
        img_rsz = Image.fromarray((img*255).astype(np.uint8))
        img_rsz = img_rsz.resize((self.size, self.size), self.inter)
        img_rsz = np.array(img_rsz).astype(np.float32) / 255

        return img_rsz


class ImageProcess:
    def __init__(self, img_shape):
        
        channel, height, width = img_shape
        if channel == 1:
            self.mean, self.std = [0.5, ], [0.5, ]
        elif channel == 3:
            # (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)   
            # [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]   
            # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            # (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            self.mean, self.std = [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]  

    def __call__(self, x, *args, **kwargs):
        image, label = x
        image = F.to_tensor(image)
        image = F.normalize(image, self.mean, self.std)

        image = image.float()
        return image, label
    

class InvImageProcess:
    def __init__(self, img_shape, to_int=True):
        channel, height, width = img_shape
        if channel == 1:
            self.mean, self.std = [0.5, ], [0.5, ]
        elif channel == 3:
            # (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)   
            # [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]   
            # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            # (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            self.mean, self.std = [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]  

        self.denormalize = Denormalize(self.mean, self.std)
        self.to_int = to_int

    def __call__(self, x, *args, **kwargs):
        image, label = x
        image = self.denormalize(image)
        image = image.detach().cpu().numpy().transpose(1, 2, 0)
        if self.to_int:
            image = (image*255).astype(np.uint8)

        return image, label
    

class Denormalize(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


class Augmentation:
    def __init__(self, img_shape=(3, 32, 32)):
        self.random_crop = transforms.RandomCrop(img_shape[1], padding=4)
        self.random_rotation = transforms.RandomRotation(15)
        self.random_horizental_flip = transforms.RandomHorizontalFlip(0.5)

    def __call__(self, x, *args, **kwargs):
        image, label = x
        # transform to PIL image
        if isinstance(image, np.ndarray):
            # image = image.transpose((1, 2, 0))
            if image.dtype in [float, np.float16, np.float32, np.float64]:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image.squeeze())
        # data augmentation
        image = self.random_crop(image)
        image = self.random_rotation(image)
        image = self.random_horizental_flip(image)

        # temp = Image.fromarray(np.uint8(image.numpy().transpose(1, 2, 0)))
        # temp.show()
        # return image.float(), label
        return image, label
    

class Augmentation2:
    def __init__(self, img_shape=(3, 32, 32), dataset=None):
        self.dataset = dataset

        self.trans_ls = [
            transforms.RandomCrop(img_shape[1], padding=4), 
            transforms.RandomRotation(15), 
        ]

        if dataset in ['mnist']:
            self.trans_ls.append(transforms.Normalize([0.5], [0.5]))
        elif dataset in ['cifar10']:
            self.trans_ls.append(transforms.RandomHorizontalFlip(0.5))
            # ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  ([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])  ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            self.trans_ls.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        else:
            pass

        self.trans = transforms.Compose(self.trans_ls)

    def __call__(self, x, *args, **kwargs):
        image, label = x
        # transform to PIL image
        if isinstance(image, np.ndarray):
            if image.dtype in [float, np.float16, np.float32, np.float64]:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image.squeeze())

        # data augmentation
        self.trans(image)

        return image, label
    

class AugmentationPost:
    def __init__(self, img_shape=(3, 32, 32)):
        self.random_crop = transforms.RandomCrop(img_shape[1], padding=4)
        self.random_rotation = transforms.RandomRotation(15)
        self.random_horizental_flip = transforms.RandomHorizontalFlip(0.5)

    def __call__(self, x, *args, **kwargs):
        image, label = x
        # data augmentation
        image = self.random_crop(image)
        image = self.random_rotation(image)
        image = self.random_horizental_flip(image)

        return image, label
    
    def apply_batch(self, batch_x, batch_y):
        # data augmentation
        batch_x = self.random_crop(batch_x)
        batch_x = self.random_rotation(batch_x)
        batch_x = self.random_horizental_flip(batch_x)

        return batch_x, batch_y


class ODProcess:
    def __init__(self, ):
        self.scaling = Scaling()
        self.rf = RandomHorizontalFlip(1)

    def __call__(self, x, *args, **kwargs):
        # TODO: image: channel_order, scaling(1/255)
        image, ann = x
        gt_boxes = ann['boxes']
        image, gt_boxes = self.scaling(image, gt_boxes)
        # image, gt_boxes = self.rf(image, gt_boxes)

        return image, gt_boxes


class Scaling:
    def __init__(self, factor=1.0, min_dim_pix=600):
        self.factor = factor
        self.min_dim_pix = min_dim_pix

    def __call__(self, image, boxes, *args, **kwargs):
        h, w = image.height, image.width
        min_edge = h if h < w else w
        factor = self.min_dim_pix / min_edge if self.min_dim_pix > min_edge else 1.0
        image = image.resize((int(w * factor), int(h * factor)), resample=Image.BILINEAR)
        for box in boxes:
            box[1] = (box[1] * factor).astype(int)
        return image, boxes


# random horizontal flip in object detection
class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, boxes):
        if random.random() < self.prob:
            # image = image.flip(-1)  # wand
            for box in boxes:
                box[1][[1, 3]] = image.width - box[1][[3, 1]]  # (y1, x1, y2, x2)
        return image.transpose(Image.FLIP_LEFT_RIGHT), boxes

