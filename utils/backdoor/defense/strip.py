'''
    STRIP: a defence against trojan attacks on deep neural networks (https://dl.acm.org/doi/abs/10.1145/3359789.3359790)

    code:
        Backdoor Toolbox (https://github.com/vtu81/backdoor-toolbox/blob/main/cleansers_tool_box/strip.py)
'''
import torch
import random
import numpy as np
from tqdm import tqdm
from sklearn import metrics


class STRIP:
    def __init__(self, img_shape, num_classes, tgt_ls=None, alpha=0.5, N=64, thres_fpr=0.05, device='cpu'):
        self.img_shape = img_shape
        self.c, self.h, self.w = img_shape
        self.num_classes = num_classes
        self.tgt_ls = tgt_ls if tgt_ls else list(range(num_classes))
        self.device = device

        self.alpha = alpha
        self.N = N
        self.thres_fpr = thres_fpr
        self.res = []
        self.config = {'type': 'STRIP', 'img_shape': img_shape, 'num_classes': num_classes, 'tgt_ls': tgt_ls,
                       'alpha': alpha, "N": N, "thres_fpr": thres_fpr}
        
    
    def __call__(self, model, ds_val, ds_clean):
        suspicious_indices = self.cleanse(model, ds_val, ds_clean)
        return suspicious_indices


    def cleanse(self, model, ds_val, ds_clean, batch_size=128, poison_rate=0.01):  
        # choose a decision boundary with the test set
        clean_entropy = []
        dl_clean = torch.utils.data.DataLoader(ds_clean, batch_size=batch_size, shuffle=False)
        for _input, _label in tqdm(dl_clean):
            _input, _label = _input.to(self.device), _label.to(self.device)
            entropies = self.check(_input, _label, model, ds_clean)
            clean_entropy.extend(entropies)
        clean_entropy = torch.FloatTensor(clean_entropy)

        clean_entropy, _ = clean_entropy.sort()
        threshold_low = float(clean_entropy[int(self.thres_fpr * len(clean_entropy))])
        threshold_high = np.inf

        # now cleanse the inspection set with the chosen boundary
        dl_inspection = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=False)
        all_entropy = []
        for _input, _label in tqdm(dl_inspection):
            _input, _label = _input.to(self.device), _label.to(self.device)
            entropies = self.check(_input, _label, model, ds_clean)
            all_entropy.extend(entropies)
        all_entropy = torch.FloatTensor(all_entropy)

        suspicious_indices = torch.logical_or(all_entropy < threshold_low, all_entropy > threshold_high).nonzero().reshape(-1)
        self.res = suspicious_indices
        print("STRIP suspicious: [{}/{}]".format(len(suspicious_indices), len(all_entropy)))  # print("STRIP suspicious indices:", suspicious_indices)

        poison_num = int(len(ds_val)*poison_rate)
        y_true = torch.cat((torch.ones(poison_num), torch.zeros(len(ds_val) - poison_num)))
        y_score = all_entropy
        y_pred = torch.logical_or(all_entropy < threshold_low, all_entropy > threshold_high)
        fpr, tpr, thresholds = metrics.roc_curve(y_true, -y_score)
        auc = metrics.auc(fpr, tpr)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        tpr, fpr = tp / (tp + fn) * 100, fp / (tn + fp) * 100
        myf1 = metrics.f1_score(y_true, y_pred)
        res = [tn, fp, fn, tp, round(auc, 6), round(tpr, 6), round(fpr, 6), round(myf1, 6)]
        print("STRIP: TPR: {:.2f}  FPR: {:.2f}  AUC: {:.4f}  F1:{}".format(tp / (tp + fn) * 100, fp / (tn + fp) * 100, auc, myf1))
        return suspicious_indices, res 


    def check(self, _input, _label, model, ds_clean):
        _list = []
        samples = list(range(len(ds_clean)))
        random.shuffle(samples)
        samples = samples[:self.N]

        with torch.no_grad():
            for i in samples:
                X, Y = ds_clean[i]
                X, Y = X.to(self.device), Y
                _test = self.superimpose(_input, X)
                entropy = self.entropy(_test, model).cpu().detach()
                _list.append(entropy)
                # _class = self.model.get_class(_test)

        return torch.stack(_list).mean(0)


    def superimpose(self, _input1, _input2):
        result = _input1 + self.alpha * _input2
        return result


    def entropy(self, input, model):
        # p = self.model.get_prob(_input)
        p = torch.nn.Softmax(dim=1)(model(input)) + 1e-8
        return (-p * p.log()).sum(1)




if __name__ == "__main__":
    # strip = STRIP(args.img_shape, args.num_classes, alpha=0.5, N=64, thres_fpr=0.05, device=device)
    # print(strip.config)
    # suspicious_indices = strip.cleanse(model, ds_inspection=ds, ds_clean=ds_vals[0])
    # print("Suspicious Indices:", suspicious_indices)
    pass


