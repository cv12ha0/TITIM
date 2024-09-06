'''
    Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering (https://arxiv.org/abs/1811.03728)

    code:
        Backdoor Toolbox (https://github.com/vtu81/backdoor-toolbox/blob/main/cleansers_tool_box/activation_clustering.py)
'''
import os
import sys
import warnings
from tqdm import tqdm

import numpy as np
import torch

from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA, FastICA
from sklearn.cluster import kmeans_plusplus, KMeans
from sklearn.metrics import silhouette_score
from sklearn.exceptions import ConvergenceWarning

from ...misc import CompactJSONEncoder, get_features

__all__ = [
    'ActivationClustering', 
]


class ActivationClustering:
    def __init__(self, img_shape, num_classes, tgt_ls=None, thres_size=0.35, thres_silhouette=0.09, decomposition='FastICA', device='cpu'):
        # warnings.warn('ActivationClustering: ignoring warnings.')
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        self.img_shape = img_shape
        self.c, self.h, self.w = img_shape
        self.num_classes = num_classes
        self.device = device

        self.tgt_ls = tgt_ls if tgt_ls else list(range(num_classes))
        self.thres_size = thres_size
        self.thres_silhouette = thres_silhouette
        self.decomposition = self.get_decomposition_func(decomposition)  # decomposition
        self.res = []
        self.scores = {'size': [], 'silhouette': []}

        self.config = {'type': 'ActivationClustering', 'img_shape': img_shape, 'num_classes': num_classes, 'tgt_ls': tgt_ls,
                       'thres_size': thres_size, 'thres_silhouette': thres_silhouette, 'decomposition': decomposition, }

    def clustering(self, ds, model, layer_name=None, batch_size=128):
        dl = torch.utils.data.DataLoader(ds, batch_size, shuffle=True, num_workers=0, drop_last=False)
        feats, labels, predicts = get_features(dl, model, layer_name, device=self.device)
        for cls_idx in range(self.num_classes):
            activations_cur = feats[torch.where(predicts == cls_idx)[0]]
            activations_cur = torch.flatten(activations_cur, 1)
            # trans_cur = self.decomposition(activations_cur.cpu().numpy())
            trans_cur = activations_cur.cpu().numpy()
            kmeans_pre = KMeans(n_clusters=2, n_init=10, random_state=0, max_iter=1000).fit_predict(trans_cur)

            # judge by cluster size
            sizes = np.bincount(kmeans_pre, minlength=2) / len(kmeans_pre)
            size_min = np.min(sizes)
            flag1 = np.any(sizes < self.thres_size)

            # judge by silhouette score
            silhouette = silhouette_score(trans_cur, kmeans_pre)
            flag2 = silhouette > self.thres_silhouette
            res_cur = 'class_idx: {}    sizes:[{:.6f} {:.6f}] min:{:.4f}  flag1: {:<5}    silhouette: {:.8f}  flag2: {:<5}'\
                      .format(cls_idx, sizes[0], sizes[1], size_min, str(flag1), silhouette, str(flag2))
            self.res.append(res_cur)
            self.scores['size'].append(size_min)
            self.scores['silhouette'].append(silhouette)
            print(res_cur)
        return self.scores

    def get_decomposition_func(self, func_name='FastICA'):
        if func_name in ['FastICA', 'fastica', 'ICA']:
            return FastICA(n_components=10, max_iter=1000, whiten="arbitrary-variance", tol=0.005)
        elif func_name in ['PCA', 'pca']:
            return PCA(n_components=10)
        elif func_name in ['KernelPCA', 'kernelpca']:
            return KernelPCA(n_components=10, max_iter=1000)
        elif func_name in ['TruncatedSVD', 'truncatedsvd']:
            return TruncatedSVD(n_components=10, n_iter=20)
        elif func_name in ['NPSVD', 'npsvd']:
            def svd(feats, n_components=10):
                feats = feats - feats.mean(0)
                _, _, V = np.linalg.svd(feats, compute_uv=True, full_matrices=True)
                axes = V[:, :n_components]
                feats_proj = np.matmul(feats, axes)
                return feats_proj
            return svd
        else:
            raise Exception('Activation Clustering: decomposition func not implemented')

    def dump_stats(self, path, info_model=None, model_dir=None):
        import json
        os.makedirs(path, exist_ok=True)
        info = {'config': self.config, 'res': self.res, 'model': info_model, 'model_dir': model_dir}

        json.dump(info, open(os.path.join(path, 'stats.json'), 'w'), cls=CompactJSONEncoder, indent=4)
        print('Activation Clustering stats saved to: ' + path)


if __name__ == "__main__":
    # activation_clustering = ActivationClustering(args.img_shape, args.num_classes, device=device)
    # activation_clustering.clustering(ds, model, layer_name='avgpool')  # layer4  conv_block3
    # activation_clustering.dump_stats(os.path.join('data', args.dataset, args.subset, 'ActivationClustering'), model.config, model_dir)
    pass


