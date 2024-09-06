import warnings
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from PIL import Image

from utils.datasets import get_dataset_pkl
from utils.misc import set_random_seed, get_device, get_nw, OfficPreProcess, ImageProcess


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    img_shape = (3, 32, 32)
    dataset = 'gtsrb'
    dataset_path = os.path.join(dataset, 'clean')
    split = 'train'
    fpath = os.path.join('./data', dataset_path)
    os.makedirs(fpath, exist_ok=True)
    print("dataset: ", dataset)

    # ds = get_dataset(dataset, split=split)
    ds = get_dataset_pkl(dataset_path, split)

    res = np.zeros((img_shape[1], img_shape[2], img_shape[0]))
    item_cnt = 1
    prog_bar = tqdm(ds, file=sys.stdout)
    for idx, item in enumerate(prog_bar, 0):
        res += item[0]
        item_cnt += 1

    if img_shape[0] == 1:
        res_img = Image.fromarray((res / item_cnt * 255).astype(np.uint8).squeeze(), mode='L')
    else:
        res_img = Image.fromarray((res / item_cnt * 255).astype(np.uint8))
    res_img.show()
    res_img.save('avg_' + dataset + '_' + split + '.jpg')


if __name__ == '__main__':
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    main()
