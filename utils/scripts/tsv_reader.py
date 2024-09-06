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



def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    # tsv reader
    fpath = 'logs/cross_patch_center.tsv'  # logs/cross_badnets_gtsrb.tsv
    title = 'resnet18_celeba8_patch_pokemon_mr0.5_0.1'
    data = pd.read_csv(fpath, sep='\t')
    test_idx = 4
    tb_prefix = 'tmr'
    print(title)

    # name parser
    ls = []
    for i in range(len(data)):
        name, asr = data.loc[i]
        args = name.split('_')

        # model, dataset, b, m, mr, ratio, epochs, tmr = args[0], args[1], int(args[3].lstrip('b')), int(args[4].lstrip('m')), float(args[5].lstrip('mr')), float(args[6]), int(args[7].lstrip('e')), float(args[8].lstrip('tmr'))
        # ls.append([model, dataset, b, m, mr, ratio, epochs, tmr, asr])
        model, dataset, b, bn, mr, ratio, epochs, tmr = args[0], args[1], int(args[3].lstrip('b')), int(args[4].lstrip('bn')), float(args[6].lstrip('mr')), float(args[7]), int(args[8].lstrip('e')), float(args[9].lstrip('tmr'))
        ls.append([model, dataset, b, bn, mr, ratio, epochs, tmr, asr])

        # model, dataset, pattern, mr, ratio, tpattern, tmr = args[0], args[1], args[3], float(args[4].lstrip('mr')), float(args[5]), args[-2].lstrip('t'), float(args[-1].lstrip('mr'))
        # ls.append([model, dataset, pattern, mr, ratio, tpattern, tmr, asr])
    
    # filter
    def func(x):
        return True
    ls = list(filter(func, ls))

    # table fmt
    col = ['name', 'tmr0.1', 'tmr0.2', 'tmr0.3', 'tmr0.4', 'tmr0.5', 'tmr0.6', 'tmr0.7', 'tmr0.8', 'tmr0.9', 'tmr1.0']
    # col = ['name', 'tmr0.01', 'tmr0.03', 'tmr0.05', 'tmr0.07', 'tmr0.1', 'tmr0.15', 'tmr0.2', 'tmr0.25', 'tmr0.3']
    col_dict = {col[i]: i for i in range(len(col))}
    tb = []

    name_set = list(set([i[test_idx] for i in ls]))
    name_set.sort()  # (reverse=True)
    print(name_set)
    for name_cur in name_set:
        tb_cur = ['{}'.format(name_cur)] + [-1] * (len(col) - 1)

        for i in filter(lambda x: x[test_idx] == name_cur, ls):
            if (tb_prefix+str(i[-2])) in col_dict.keys():
                tb_cur[col_dict[tb_prefix+str(i[-2])]] = i[-1]
        tb.append(tb_cur)
    print(tabulate(tb, col, tablefmt='pipe'))


    # y_r = np.repeat(y, [10], axis=1)
    x_r = [i.lstrip(tb_prefix) for i in col[1:]]
    y_r = [tb[i][0].replace('\\', '') for i in range(len(tb))]
    data = np.array([tb[i][1:] for i in range(len(tb))])
    heatmap(data, x_r, y_r, fpath=title+'.pdf')

def heatmap(data, x_r, y_r, dpi=500, title='', fpath='heatmap.png'):
    ax = sns.heatmap(pd.DataFrame(data, columns=x_r, index=y_r), annot=True, annot_kws={'size': 6}, fmt='.2f', vmax=100., vmin=0.,
                     xticklabels=True, yticklabels=True, square=True, cmap="Reds")  # cmap="YlGnBu"
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    # plt.xticks(fontsize=6)
    # plt.yticks(fontsize=6)
    plt.title(title)  # badnets_gtsrb_r0.2
    plt.savefig('heatmaps/'+fpath, dpi=dpi, bbox_inches='tight')
    # plt.cla()
    plt.clf()





if __name__ == '__main__':
    main()
