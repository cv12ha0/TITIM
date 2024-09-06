'''
    Spectral Signatures in Backdoor Attacks (https://proceedings.neurips.cc/paper/2018/hash/280cf18baf4311c92aa5a042336587d3-Abstract.html)

    code:
        Backdoor Toolbox (https://github.com/vtu81/backdoor-toolbox/blob/main/cleansers_tool_box/spectral_signature.py)
        Sleeper Agent (https://github.com/hsouri/Sleeper-Agent/blob/master/forest/filtering_defenses.py)
'''
import os
import sys
from tqdm import tqdm

import numpy as np
import torch

from ...misc import CompactJSONEncoder, get_features

__all__ = ['SpectralSignature']


class SpectralSignature:
    def __init__(self, img_shape, num_classes, tgt_ls=None, device='cpu'):
        self.img_shape = img_shape
        self.c, self.h, self.w = img_shape
        self.num_classes = num_classes
        self.device = device

        self.tgt_ls = tgt_ls if tgt_ls else list(range(num_classes))
        self.suspicious_indices = []
        self.config = {'type': 'SpectralSignature', 'img_shape': img_shape, 'num_classes': num_classes, 'tgt_ls': tgt_ls, }
        

    def filter(self, ds, model, layer_name=None, batch_size=128, poison_rate=0.01):
        num_poisons_expected = poison_rate * len(ds) * 1.5
        suspicious_indices = []
        dl = torch.utils.data.DataLoader(ds, batch_size, shuffle=False, num_workers=0, drop_last=False)
        feats, labels, predicts = get_features(dl, model, layer_name, device=self.device)

        for cls_idx in self.tgt_ls:
            indices_cur = torch.where(predicts == cls_idx)[0]
            feats_cur = feats[indices_cur]
            feats_mean = torch.mean(feats_cur, dim=0)
            feats_cur -= feats_mean

            _, _, V = torch.svd(feats_cur, compute_uv=True, some=False)
            vec = V[:, 0]
            vals = []
            for i in range(feats_cur.shape[0]):
                vals.append(torch.dot(feats_cur[i], vec).pow(2))
            
            k = min(int(num_poisons_expected), len(vals)//2)
            _, indices = torch.topk(torch.tensor(vals), k)
            for i in indices:
                suspicious_indices.append(indices_cur[i].item())
        suspicious_indices.sort()

        # TODO: succ_rate calculation
        num_checked = np.sum(np.array(indices) < int(len(ds) * poison_rate))
        num_poison = torch.sum(indices_cur < int(len(ds) * poison_rate)).item()
        succ_rate = num_checked / num_poison
        succ_rate2 = num_checked/(len(ds) * poison_rate)
        print('SS: succ:{:.2f} [{}/{}]    succ2:{:.2f} [{}/{}]'
              .format(succ_rate, num_checked, num_poison, succ_rate2, num_checked, int(len(ds) * poison_rate)))
        self.suspicious_indices = suspicious_indices
        return suspicious_indices, [round(succ_rate, 4), num_checked, num_poison, round(succ_rate2, 6), num_checked, int(len(ds) * poison_rate)]

    def dump_stats(self, path, info_model=None, model_dir=None):
        import json
        os.makedirs(path, exist_ok=True)
        info = {'config': self.config, 'model': info_model, 'model_dir': model_dir, 'suspicious_indices': self.suspicious_indices}

        json.dump(info, open(os.path.join(path, 'stats.json'), 'w'), cls=CompactJSONEncoder, indent=4)
        print('Spectral Signature stats saved to: ' + path)



if __name__ == "__main__":
    # ss = SpectralSignature(args.img_shape, args.num_classes, tgt_ls=[0], device=device)
    # ss.filter(ds, model, layer_name='avgpool')  # layer4  conv_block3  avgpool
    # ss.dump_stats(os.path.join('data', args.dataset, args.subset, 'SpectralSignature'), model.config, model_dir)
    pass


