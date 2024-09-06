import os
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import torch
from tqdm import tqdm

from ..datasets import get_num_classes, get_cls_names, get_dataset_pkl
from utils.misc import ImageProcess, InvImageProcess

__all__ = [
    'tsne',
    'confusion_matrix',
    'grad_cam',
    'get_features',
    'get_features2',

    'TSNE',
    'UMAP',
]


def tsne(features, labels, title='', fpath='tsne.png'):
    # from sklearn.manifold import TSNE
    from openTSNE import TSNE

    # # normalization
    # x_min, x_max = np.min(tsne_features, 0), np.max(tsne_features, 0)
    # tsne_features = (tsne_features - x_min) / (x_max - x_min)

    tsne = TSNE(n_components=2, init='random', random_state=0, verbose=0)
    data = tsne.fit(features)  # tsne.fit_transform(features)

    plt.figure(dpi=300, figsize=(15, 10))
    # ax = sns.relplot(data=pd.DataFrame(data_full, columns=['x', 'y', 'labels', 'pres']), x='x', y='y', hue='labels')
    # ax = sns.scatterplot(data_full, x='x', y='y', hue='labels', s=4)
    ax = sns.relplot(x=data[:, 0], y=data[:, 1], hue=labels, s=4, alpha=0.7, 
                     palette=sns.color_palette(hex, 10), 
                     markers={"c0": ".", "c1": ".", "c2": ".", "c3": ".", "c4": "."}, 
                     ).set(title=title)
    # plt_sne.legend(loc="lower right")

    plt.savefig(fpath, dpi=300)
    # plt.clf()
    return


def umap_(features, labels, fpath='umap.png'):
    from umap import UMAP as UMAP_ 
    umap = UMAP_(n_components=2, init='spectral', random_state=0, metric="euclidean")
    data = umap.fit_transform(features)

    plt.figure(dpi=300, figsize=(15, 10))
    # ax = sns.relplot(data=pd.DataFrame(data_full, columns=['x', 'y', 'labels', 'pres']), x='x', y='y', hue='labels')
    # ax = sns.scatterplot(data_full, x='x', y='y', hue='labels', s=4)
    ax = sns.relplot(x=data[:, 0], y=data[:, 1], hue=labels, s=4, alpha=0.7)

    plt.savefig(fpath, dpi=300)
    # plt.clf()
    return


def confusion_matrix(pres, labels, dataset, num_classes=None, use_cls_name=False, normalize=True, 
                     title='', fpath='confussion_matrix.png'):
    num_classes = get_num_classes(dataset, num_classes)
    if use_cls_name:
        cls_names = get_cls_names(dataset, num_classes=num_classes)
    else:
        cls_names = [str(i) for i in range(num_classes)]

    # calculate confussion matrix
    cm = np.zeros((num_classes, num_classes))
    for i in range(len(pres)):
        cm[labels[i], pres[i]] += 1

    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-24)
    else:
        cm = cm.astype("int")

    # plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
        xticklabels=cls_names, yticklabels=cls_names, 
        title=title,
        ylabel="True label",  # labels  True label
        xlabel="Predicted label",  # pres  Predicted label
    )
    if use_cls_name:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black", )
    fig.tight_layout()
    plt.xlim(-0.5, num_classes - 0.5)
    plt.ylim(num_classes - 0.5, -0.5)
    plt.tight_layout()

    plt.savefig(fpath, dpi=500, bbox_inches='tight')
    return


def grad_cam(inputs, targets, model, target_layers, img_shape=None):
    from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    from .transforms import ImageProcess
    if img_shape is None:
        img_shape = [inputs[0].shape[2], inputs[0].shape[0], inputs[0].shape[1]]
    image_process = ImageProcess(np.array(img_shape))
    inputs_tensor = torch.stack([image_process([i, 0])[0] for i in inputs])
    targets = [ClassifierOutputTarget(i) for i in targets]
    cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=inputs_tensor, targets=targets)
    visualizations = [show_cam_on_image(inputs[i].astype(float) / 255, grayscale_cam[i, :], use_rgb=True) for i in range(len(inputs))]
    model_outputs = cam.outputs
    return visualizations


def grad_cam_one(input, target, model, target_layers, img_shape=None, fpath='gradcam.png'):
    vis = grad_cam([input], [target], model, target_layers, img_shape)
    cv2.imwrite(fpath, cv2.cvtColor(vis[0], cv2.COLOR_RGB2BGR))





class TSNE:
    def __init__(self, normalize=False, n_components=2, init='random', random_state=0, verbose=0, *args, **kargs) -> None:
        # from sklearn.manifold import TSNE
        from openTSNE import TSNE
        # self.tsne = TSNE(n_components=2, init='random', random_state=0, verbose=0)
        self.tsne = TSNE(n_components=n_components, init=init, random_state=random_state, verbose=verbose, **kargs)
        self.normalize = normalize


    def plot(self, features, labels, title='', fpath='tsne.png'):
        data = self.tsne.fit_transform(features)

        if self.normalize:
            x_min, x_max = np.min(features, 0), np.max(features, 0)
            features = (features - x_min) / (x_max - x_min)

        plt.figure(dpi=300, figsize=(15, 10))
        # ax = sns.relplot(data=pd.DataFrame(data_full, columns=['x', 'y', 'labels', 'pres']), x='x', y='y', hue='labels')
        # ax = sns.scatterplot(data_full, x='x', y='y', hue='labels', s=4)
        ax = sns.relplot(x=data[:, 0], y=data[:, 1], hue=labels, s=4, alpha=0.7, 
                         palette=sns.color_palette(hex, 10), 
                         markers={"c0": ".", "c1": ".", "c2": ".", "c3": ".", "c4": "."}, 
                         ).set(title=title)
        # plt_sne.legend(loc="lower right")

        plt.savefig(fpath, dpi=300)
        # plt.clf()
        return


class UMAP:
    def __init__(self, normalize=False, n_components=2, init='random', random_state=0, metric="euclidean", *args, **kargs) -> None:
        from umap import UMAP
        self.umap = UMAP(n_components=n_components, init=init, random_state=random_state, metric=metric, **kargs)
        self.normalize = normalize

    def plot(self, features, labels, title='', fpath='tsne.png'):
        data = self.umap.fit_transform(features)

        plt.figure(dpi=300, figsize=(15, 10))
        # ax = sns.relplot(data=pd.DataFrame(data_full, columns=['x', 'y', 'labels', 'pres']), x='x', y='y', hue='labels')
        # ax = sns.scatterplot(data_full, x='x', y='y', hue='labels', s=4)
        ax = sns.relplot(x=data[:, 0], y=data[:, 1], hue=labels, s=4, alpha=0.7)

        plt.savefig(fpath, dpi=300)
        # plt.clf()
        return




'''
    utils
'''
def load_model(model_dir, model_name, device):
    model_dir = os.path.join('res', model_dir, 'weights', model_name+'.pth')
    model = torch.load(model_dir, map_location=device)['model']  # .to(device)
    print("model: ", model_dir)
    return model


def load_dl(dataset, subset, split, img_shape, batch_size, num_workers):
    ds = get_dataset_pkl(os.path.join(dataset, subset), split, )
    # preprocess
    image_process = ImageProcess(img_shape)
    ds = ds.map(image_process)
    dl = torch.utils.data.DataLoader(ds, batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    print("dataset: ", os.path.join(dataset, subset, split))
    return dl


def set_feature_hook(model, feats_hook, layer_name='avgpool', loc='out'):
    def hook(module, fea_in, fea_out):
        if loc in ['in']:
            feats_hook.append(fea_in.detach())
        elif loc in ['out']:
            feats_hook.append(fea_out.detach())
        return None
    
    for (name, module) in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook=hook)
            return handle
    raise Exception('set_feature_hook(): layer[{}] not found.'.format(layer_name))


def get_features(dl, model, layer_name='avgpool', device='cpu', disable=False):
    feats, labels, predicts = [], [], []
    model.eval()
    set_feature_hook(model, feats, layer_name)

    with torch.no_grad():
        prog_bar = tqdm(dl, file=sys.stdout, leave=True, disable=disable)
        for step, (batch_x, batch_y) in enumerate(prog_bar):
            pre = model(batch_x.to(device))
            pre = torch.argmax(pre, 1)
            labels.append(batch_y)
            predicts.append(pre)
    feats = torch.flatten(torch.cat(feats, 0), 1)
    labels = torch.cat(labels, 0).to(device)
    predicts = torch.cat(predicts, 0)
    return feats, labels, predicts


def get_features2(dl, model, layer_name='avgpool', loc='out', flatten=True, device='cpu', disable=False):
    feats, labels, predicts = [], [], []
    model.eval()
    set_feature_hook(model, feats, layer_name, loc)

    with torch.no_grad():
        prog_bar = tqdm(dl, file=sys.stdout, leave=True, disable=disable)
        for step, (batch_x, batch_y) in enumerate(prog_bar):
            pre = model(batch_x.to(device))
            pre = torch.argmax(pre, 1)
            labels.append(batch_y)
            predicts.append(pre)
    feats = torch.cat(feats, 0)
    labels = torch.cat(labels, 0).to(device)
    predicts = torch.cat(predicts, 0)
    if flatten:
        feats = torch.flatten(feats, 1)
    return feats, labels, predicts

