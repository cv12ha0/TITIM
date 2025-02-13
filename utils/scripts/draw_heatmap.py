import argparse
import warnings
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
# from scipy.interpolate import griddata


def main(args):
    warnings.filterwarnings("ignore", category=UserWarning)
    os.makedirs('heatmaps', exist_ok=True)
    fpath = 'logs/'+args.name+'.tsv'  # cross_badnets_mixmr0.1_0.05_resnet18
    title = args.name  # resnet18_celeba8_patch_pokemon_mr0.5_0.1
    data = pd.read_csv(fpath, sep='\t')
    N = int(np.sqrt(len(data))) if args.N is None else args.N  # 10
    data = np.array(data.iloc[:, 1]).reshape(N, N)

    # y_r = np.repeat(y, [' '], axis=1)
    x_r = [''] * 10
    # x_r = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # , 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0
    # x_r = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    # y_r = ['mr0.1', 'mr0.2', 'mr0.3', 'mr0.4', 'mr0.5', 'mr0.6', 'mr0.7', 'mr0.8', 'mr0.9', 'mr1.0']
    # x_r = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3]
    # x_r = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # x_r = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # x_r = [8, 10, 12, 14, 16, 18, 20, 22, 24]
    # x_r = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
    # x_r = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    # x_r = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    # x_r = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    # x_r = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
    y_r = x_r
    print('fpath: ', fpath)
    heatmap(data, x_r, y_r, fpath=title+'.pdf')  # wanet_cr0_k2_r0.1    title=title,    fpath='heatmap_'+title+'.png'

def heatmap(data, x_r, y_r, dpi=500, title='', fpath='heatmap.png'):
    # c = plt.pcolormesh(x_r, y_r, np.array([tb[i][1:] for i in range(10)]), cmap='viridis_r', shading='gouraud')  # smoothed
    # c = plt.pcolormesh(x_r, y_r, data, cmap='viridis_r')  # common
    ax = sns.heatmap(pd.DataFrame(data, columns=x_r, index=y_r), annot=True, annot_kws={'size': 6}, fmt='.2f', vmax=100., vmin=0.,
                     xticklabels=True, yticklabels=True, square=True, cmap="Reds")  # cmap="YlGnBu"
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    plt.title(title)
    plt.savefig('heatmaps/'+fpath, dpi=dpi, bbox_inches='tight')
    print('heatmap saved to  heatmaps/'+fpath)
    plt.clf()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='file name in logs/', default='')
    parser.add_argument('--N', type=int, help='num of cols / rows', default=None)
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)

