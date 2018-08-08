import torch
import numpy as np
from skimage import draw
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F


class RunningAverage:
    def __init__(self):
        self.iter = 0
        self.avg = 0.0

    def update(self, x):
        self.avg = (self.avg * self.iter + x) / (self.iter + 1)
        self.iter += 1

    def __str__(self):
        if self.iter == 0:
            return '-'
        return f'{self.avg:.4f}'


class Annotation:

    @staticmethod
    def hflip(ann):
        kpts = np.float32(ann['kpts'])
        kpts[:, 1] -= 0.5
        kpts[:, 1] *= -1
        kpts[:, 1] += 0.5
        ann = ann.copy()
        ann['kpts'] = kpts.tolist()
        return ann

    @staticmethod
    def extend(ann, edges, n_points):
        ann = ann.copy()
        kpts = np.float32(ann['kpts'])
        tags = np.int32(ann['tags'])
        idxs = np.int32(ann['idxs'])

        for pid in range(ann['n_people']):
            for (s, t) in edges:
                kpt_s = kpts[(tags == pid) & (idxs == s)]
                kpt_t = kpts[(tags == pid) & (idxs == t)]
                if kpt_s.size == 0 or kpt_t.size == 0:
                    continue
                unit = (kpt_t - kpt_s) / (n_points - 1)
                for i in range(1, n_points - 1):
                    kpt = kpt_s + unit * i
                    ann['kpts'].append(kpt.ravel().tolist())
                    ann['tags'].append(pid)
                    ann['idxs'].append(-1)

        return ann

    @staticmethod
    def draw(img, ann):
        colors = plt.cm.tab20(np.linspace(0.0, 1.0, 20))[:, :3]
        colors = torch.FloatTensor(colors).unsqueeze(2)

        _, h, w = img.size()
        img = img.clone()
        img = img * 0.4
        size = torch.FloatTensor([h, w])
        kpts = torch.FloatTensor(ann['kpts'])
        tags = torch.LongTensor(ann['tags'])
        kpts = torch.floor(kpts * size).long()
        for pid in range(ann['n_people']):
            for (r, c) in kpts[tags == pid]:
                rr, cc = draw.circle(r, c, 1.5, shape=size)
                img[:, rr, cc] = colors[pid % 20]

        return img
