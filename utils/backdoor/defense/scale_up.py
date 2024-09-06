'''
    SCALE-UP: An Efficient Black-box Input-level Backdoor Detection via Analyzing Scaled Prediction Consistency (https://arxiv.org/abs/2302.03251)

    code:
        Official (https://github.com/JunfengGo/SCALE-UP)
        Backdoor Toolbox (https://github.com/vtu81/backdoor-toolbox/blob/main/other_defenses_tool_box/scale_up.py)
        BackdoorBox (https://github.com/THUYimingLi/BackdoorBox/blob/main/core/defenses/SCALE_UP.py)
'''
import os
import sys
import warnings
from tqdm import tqdm
from sklearn import metrics

import numpy as np
import torch
import torchvision

from ...misc import CompactJSONEncoder, Denormalize

__all__ = [
    'ScaleUp', 
]


class ScaleUp:
    def __init__(self, img_shape, num_classes, scale_set=None, thres=0.5, device='cpu'):
        self.img_shape = img_shape
        self.c, self.h, self.w = img_shape
        self.num_classes = num_classes
        self.device = device

        self.scale_set = scale_set if scale_set is not None else [3, 5, 7, 9, 11]
        self.thres = thres
        self.denormalizer = Denormalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        self.normalizer = torchvision.transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        self.mean, self.std = 0, 1

        # self.scores = {'size': [], 'silhouette': []}

        self.config = {'type': 'ScaleUp', 'img_shape': img_shape, 'num_classes': num_classes, 
                       'scale_set': self.scale_set, 'thres': self.thres}



    def init_spc_norm(self, ds, model, device='cpu'):
        dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)
        model.eval()
        total_spc = []
        for idx, batch in enumerate(dl):
            clean_img, labels = batch[0].to(device), batch[1].to(device)
            scaled_imgs, scaled_labels = [], []
            for scale in self.scale_set:
                scaled_imgs.append(self.normalizer(torch.clip(self.denormalizer(clean_img) * scale, 0.0, 1.0)))
            for scaled_img in scaled_imgs:
                scaled_label = torch.argmax(model(scaled_img), dim=1)
                scaled_labels.append(scaled_label)

            # compute the SPC Value
            spc = torch.zeros(labels.shape).to(device)
            for scaled_label in scaled_labels:
                spc += scaled_label == labels
            spc /= len(self.scale_set)
            total_spc.append(spc)
        total_spc = torch.cat(total_spc)

        self.mean, self.std = torch.mean(total_spc).item(), torch.std(total_spc).item()


    def detect(self, ds, model, poison_rate=0.01, device='cpu'):
        data_loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)
        model.eval()
        all_spc_score = []
        pred_correct_mask = []

        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                imgs = batch[0].to(device)
                labels = batch[1].to(device)
                preds_original = torch.argmax(model(imgs), dim=1)
                mask = torch.eq(labels, preds_original)  # only look at those samples that successfully attack the DNN
                pred_correct_mask.append(mask)

                scaled_imgs, scaled_labels = [], []
                for scale in self.scale_set:
                    scaled_imgs.append(self.normalizer(torch.clip(self.denormalizer(imgs) * scale, 0.0, 1.0)))
                for scale_img in scaled_imgs:
                    scale_label = torch.argmax(model(scale_img), dim=1)
                    scaled_labels.append(scale_label)
                
                spc_score = torch.zeros(labels.shape).to(device)
                for scale_label in scaled_labels:
                    spc_score += scale_label == preds_original
                spc_score /= len(self.scale_set)
                # normalize by benign samples
                spc_score = (spc_score - self.mean) / self.std
                all_spc_score.append(spc_score)
        
        all_spc_score = torch.cat(all_spc_score, dim=0).cpu()
        pred_correct_mask = torch.cat(pred_correct_mask, dim=0)
        # all_spc_score = all_spc_score[pred_correct_mask]

        # cal succ / AUROC
        poison_num = int(len(ds)*poison_rate)
        y_true = torch.cat((torch.ones(poison_num), torch.zeros(len(ds) - poison_num)))
        y_score = all_spc_score
        y_pred = (y_score > self.thres)
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        auc = metrics.auc(fpr, tpr)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        tpr, fpr = tp / (tp + fn) * 100, fp / (tn + fp) * 100
        myf1 = metrics.f1_score(y_true, y_pred)
        res = [tn, fp, fn, tp, auc, tpr, fpr, myf1]
        print("[All sample] TPR: {:.2f}  FPR: {:.2f}  AUC: {:.4f}  F1:{}".format(tp / (tp + fn) * 100, fp / (tn + fp) * 100, auc, myf1))
        # print("TPR: {:.2f}".format(tp / (tp + fn) * 100))
        # print("FPR: {:.2f}".format(fp / (tn + fp) * 100))
        # print("AUC: {:.4f}".format(auc))
        # print(f"f1 score: {myf1}")
        pred_correct_mask = pred_correct_mask.cpu()
        poison_num = (torch.where(pred_correct_mask)[0] < poison_num).sum()
        y_true = torch.cat((torch.ones(poison_num), torch.zeros(pred_correct_mask.sum() - poison_num)))
        y_score = all_spc_score[pred_correct_mask]
        y_pred = (y_score > self.thres)
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        auc = metrics.auc(fpr, tpr)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        tpr, fpr = tp / (tp + fn) * 100, fp / (tn + fp) * 100
        myf1 = metrics.f1_score(y_true, y_pred)
        res += [tn, fp, fn, tp, auc, tpr, fpr, myf1]
        print("[Filtered]   TPR: {:.2f}  FPR: {:.2f}  AUC: {:.4f}  F1:{}".format(tp / (tp + fn) * 100, fp / (tn + fp) * 100, auc, myf1))
        
        # return all_spc_score, tn, fp, fn, tp, tpr, fpr, auc, myf1
        return res



    def dump_stats(self, path, info_model=None, model_dir=None):
        import json
        os.makedirs(path, exist_ok=True)
        info = {'config': self.config, 'res': self.res, 'model': info_model, 'model_dir': model_dir}

        json.dump(info, open(os.path.join(path, 'stats.json'), 'w'), cls=CompactJSONEncoder, indent=4)
        print('Scale Up stats saved to: ' + path)


if __name__ == "__main__":
    pass


