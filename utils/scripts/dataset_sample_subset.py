import os
import sys
import json
import pandas as pd

print(os.getcwd())
sys.path.append(os.getcwd())
from utils.datasets import get_dataset, get_img_shape, get_num_classes, sample_per_class, dump_ds_pd


def main():
    dataset = 'cifar10'
    num_per_class = 100
    num_classes = get_num_classes(dataset)
    img_shape = get_img_shape(dataset)
    subset = 'clean'
    split = 'test'
    fname = split + '_sample_' + str(num_per_class)
    output_dir = os.path.join('data', dataset, subset, )
    # os.makedirs(output_dir, exist_ok=True)
    print("dataset:", dataset, '  subset:', subset, '  split:', split)

    ds = get_dataset(os.path.join(dataset, subset), split, fmt='pkl')

    # transforms
    ds = sample_per_class(ds, num_classes, num_per_class)

    # save
    info = {fname: {'dataset': dataset, 'sample_per_class': num_per_class}}
    dump_ds_pd(ds.data, output_dir, fname, info, columns=['image', 'label'], use_ndarray2pil=False)


if __name__ == '__main__':
    main()
