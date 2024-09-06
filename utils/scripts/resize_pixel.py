import os
import sys
import numpy as np
from PIL import Image
import cv2


def main():
    fpath = 'temp/gtsrb'  # 'temp2'
    # img = Image.open('test02.png')
    # img_rsz = resize_pix(img, 10)

    # cv2.resize
    
    for dirpath, dirnames, filenames in os.walk(fpath):
        for filename in filenames:
            if filename.endswith('.png'):
                print('resizing: ', os.path.join(dirpath, filename))
                fname_cur = dirpath+'/'+filename
                img = Image.open(fname_cur)
                img_rsz = resize_pix(img, 10)
                Image.fromarray(img_rsz).save(fname_cur)



def resize_pix(img, factor=10):
    img = np.atleast_3d(np.array(img))
    shape = [img.shape[0]*factor, img.shape[1]*factor, img.shape[2]]
    img_rsz = np.zeros(shape, dtype=np.uint8)

    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            for c in range(img.shape[2]):
                img_rsz[factor*h: factor*(h+1), factor*w: factor*(w+1), c] = img[h, w, c]

    # img_rsz = np.tile(img, [factor, factor, 1])
    img_rsz = img_rsz.squeeze()
    return img_rsz




if __name__ == '__main__':
    main()
