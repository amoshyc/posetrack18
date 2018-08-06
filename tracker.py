from random import randint
from pathlib import Path
from datetime import datetime

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import *
from torchvision.utils import save_image
from torch.utils.data.dataset import Subset, ConcatDataset

import numpy as np
from tqdm import tqdm
from skimage import feature
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from posetrack import PoseTrack
from util import RunningAverage

class SiameseTagLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ebd1_batch, ebd2_batch, ann1_batch, ann2_batch):
        (B, D, H, W), device = ebd1_batch.size(), ebd1_batch.device
        losses = torch.zeros(B, dtype=torch.float, device=device)

        for i in range(B):
            ebd1, ebd2 = ebd1_batch[i], ebd2_batch[i]
            ann1, ann2 = ann1_batch[i], ann2_batch[i]
            true_ebd, pred_ebd = \
                self.extract(ebd1, ebd2, ann1, ann2, [H, W], device)
            K = len(true_ebd)

            A = true_ebd.expand(K, K)                   # (K, K)
            B = A.t()                                   # (K, K)
            true_similarity = (A == B).float()          # (K, K)

            A = pred_ebd.unsqueeze(1)                   # (D, 1, K)
            A = A.expand(D, K, K)                       # (D, K, K)
            B = pred_ebd.unsqueeze(2)                   # (D, K, 1)
            B = B.expand(D, K, K)                       # (D, K, K)
            expo = ((A - B)**2).mean(dim=0)             # (K, K)
            pred_similarity = 2 / (1 + torch.exp(expo))

            losses[i] = ((true_similarity - pred_similarity)**2).mean()

        return losses.mean()

    def extract(self, ebd1, ebd2, ann1, ann2, size, device):
        size = torch.FloatTensor(size).to(device)
        kpts1 = torch.FloatTensor(ann1['kpts']).to(device)
        kpts2 = torch.FloatTensor(ann2['kpts']).to(device)
        mask1_r = (0.0 <= kpts1[:, 0]) & (kpts1[:, 0] < 1.0)
        mask1_c = (0.0 <= kpts1[:, 1]) & (kpts1[:, 1] < 1.0)
        mask1 = mask1_r & mask1_c
        mask2_r = (0.0 <= kpts2[:, 0]) & (kpts2[:, 0] < 1.0)
        mask2_c = (0.0 <= kpts2[:, 1]) & (kpts2[:, 1] < 1.0)
        mask2 = mask2_r & mask2_c

        true_ebd1 = torch.LongTensor(ann1['tags']).to(device)
        true_ebd2 = torch.LongTensor(ann2['tags']).to(device)
        true_ebd1 = true_ebd1[mask1]
        true_ebd2 = true_ebd2[mask2]
        true_ebd = torch.cat([true_ebd1, true_ebd2], dim=0)     # (K,)

        kpts1 = torch.floor(kpts1[mask1] * size).long()
        kpts2 = torch.floor(kpts2[mask2] * size).long()
        pred_ebd1 = ebd1[:, kpts1[:, 0], kpts1[:, 1]]
        pred_ebd2 = ebd2[:, kpts2[:, 0], kpts2[:, 1]]
        pred_ebd = torch.cat([pred_ebd1, pred_ebd2], dim=1)     # (D, K)

        return true_ebd, pred_ebd


class PoseTracker:
    def __init__(self, log_dir, device):
        self.device = device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        self.model = Backbone().to(device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = SiameseTagLoss()

    def fit(self, train_set, valid_set, vis_set, epoch=100):
        self.train_loader = DataLoader(train_set, batch_size=16,
            shuffle=True, collate_fn=PoseTrack.collate_fn, num_workers=4)
        self.valid_loader = DataLoader(valid_set, batch_size=16,
            shuffle=False, collate_fn=PoseTrack.collate_fn, num_workers=4)
        self.vis_loader = DataLoader(vis_set, batch_size=16,
            shuffle=False, collate_fn=PoseTrack.collate_fn, num_workers=4)

        self.log = pd.DataFrame()
        for self.ep in range(epoch):
            self.epoch_dir = (self.log_dir / f'{self.ep:03d}')
            self.epoch_dir.mkdir()
            self.msg = dict()

            tqdm_args = {
                'total': len(train_set) + len(valid_set),
                'desc': f'Epoch {self.ep:03d}',
                'ascii': True,
            }
            with tqdm(**tqdm_args) as self.pbar:
                self._train()
                with torch.no_grad():
                    self._valid()
                    self._vis()
                    self._log()

    def _train(self):
        self.msg.update({
            'loss': RunningAverage(),
        })
        self.model.train()
        for img1_batch, img2_batch, \
            ann1_batch, ann2_batch in iter(self.train_loader):
            img1_batch = img1_batch.to(self.device)
            img2_batch = img2_batch.to(self.device)

            self.optim.zero_grad()
            ebd1_batch = self.model(img1_batch)
            ebd2_batch = self.model(img2_batch)
            loss = self.criterion(ebd1_batch, ebd2_batch, ann1_batch, ann2_batch)
            loss.backward()
            self.optim.step()

            self.msg['loss'].update(loss.item())
            self.pbar.set_postfix(self.msg)
            self.pbar.update(len(img1_batch))

    def _valid(self):
        self.msg.update({
            'val_loss': RunningAverage(),
        })
        self.model.eval()
        for img1_batch, img2_batch, \
            ann1_batch, ann2_batch in iter(self.valid_loader):
            img1_batch = img1_batch.to(self.device)
            img2_batch = img2_batch.to(self.device)

            ebd1_batch = self.model(img1_batch)
            ebd2_batch = self.model(img2_batch)
            loss = self.criterion(ebd1_batch, ebd2_batch, ann1_batch, ann2_batch)

            self.msg['val_loss'].update(loss.item())
            self.pbar.update(len(img1_batch))
        self.pbar.set_postfix(self.msg)

    def _vis(self):
        self.model.eval()
        idx = 0
        for img1_batch, img2_batch, \
            ann1_batch, ann2_batch in iter(self.vis_loader):
            img1_batch = img1_batch.to(self.device)
            img2_batch = img2_batch.to(self.device)
            _, _, H, W = img1_batch.size()
            ebd1_batch = self.model(img1_batch)
            ebd2_batch = self.model(img2_batch)
            ebd1_batch = F.upsample(ebd1_batch, (H, W))
            ebd2_batch = F.upsample(ebd2_batch, (H, W))
            ebd1_batch = ebd1_batch * 0.45 + 0.5
            ebd2_batch = ebd2_batch * 0.45 + 0.5

            for img1, img2, ebd1, ebd2 in \
                zip(img1_batch, img2_batch, ebd1_batch, ebd2_batch):
                vis1 = img1 * 0.5 + ebd1 * 0.5
                vis2 = img2 * 0.5 + ebd2 * 0.5
                path = f'{self.epoch_dir}/{idx:05d}.jpg'
                save_image([img1, ebd1, vis1, img2, ebd2, vis2], path, nrow=3, pad_value=1)
                idx += 1

    def _log(self):
        new_row = {k: v.avg for k, v in self.msg.items()}
        self.log = self.log.append(new_row, ignore_index=True)
        self.log.to_csv(str(self.log_dir / 'log.csv'))
        # plot loss
        fig, ax = plt.subplots(dpi=100)
        self.log[['loss', 'val_loss']].plot(ax=ax)
        fig.tight_layout()
        fig.savefig(str(self.log_dir / 'loss.jpg'))
        plt.close()  # Close plot to prevent RE
        # model
        torch.save(self.model, str(self.epoch_dir / 'model.pth'))



if __name__ == '__main__':
    root_dir = '/store/PoseTrack2017/posetrack_data/'

    PTtrain = PoseTrack(root_dir, './pt17/train/', (256, 256))
    PTvalid = PoseTrack(root_dir, './pt17/valid/', (256, 256))
    train_vis = [randint(0, len(PTtrain) - 1) for _ in range(50)]
    valid_vis = [randint(0, len(PTvalid) - 1) for _ in range(50)]
    PTvis = ConcatDataset([
        Subset(PTtrain, train_vis),
        Subset(PTvalid, valid_vis)
    ])
    print('#Train:', len(PTtrain))
    print('#Valid:', len(PTvalid))
    print('Train Vis:', train_vis)
    print('Valid Vis:', valid_vis)

    log_dir = Path(f'./log/{datetime.now():%m-%d %H:%M:%S}/')
    device = torch.device('cuda')
    tracker = PoseTracker(log_dir, device)
    tracker.fit(PTtrain, PTvalid, PTvis, epoch=50)
