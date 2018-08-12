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
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA

from util import RunningAverage
from util import Annotation as A
from util import pca
from coco import COCOKeypoint

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        self.pre = nn.Sequential(
            nn.BatchNorm2d(3),
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
        )
        self.up1 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear'),
            # nn.Conv2d(2048, 1024, (3, 3), padding=1),
            nn.ConvTranspose2d(2048, 1024, (2, 2), stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.up2 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear'),
            # nn.Conv2d(1024, 512, (3, 3), padding=1),
            nn.ConvTranspose2d(1024, 512, (2, 2), stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.up3 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear'),
            # nn.Conv2d(512, 256, (3, 3), padding=1),
            nn.ConvTranspose2d(512, 256, (2, 2), stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.post = nn.Sequential(
            nn.Conv2d(256, 32, (1, 1)),
            nn.BatchNorm2d(32),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.pre(x)

        z1 = self.resnet.layer1(x)
        z2 = self.resnet.layer2(z1)
        z3 = self.resnet.layer3(z2)
        z4 = self.resnet.layer4(z3)
        z3 = self.up1(z4) + z3
        z2 = self.up2(z3) + z2
        z1 = self.up3(z2) + z1

        x = self.post(z1)
        return x


class TagLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ebd_batch, ann_batch):
        (B, D, H, W), device = ebd_batch.size(), ebd_batch.device
        losses = torch.zeros(B, dtype=torch.float, device=device)

        for i in range(B):
            ebd = ebd_batch[i]
            ann = ann_batch[i]
            true_ebd, pred_ebd = self.extract(ebd, ann, [H, W], device)
            K = len(true_ebd)

            A = true_ebd.expand(K, K)  # (K, K)
            B = A.t()  # (K, K)
            true_similarity = (A == B).float()  # (K, K)

            A = pred_ebd.unsqueeze(1)  # (D, 1, K)
            A = A.expand(D, K, K)  # (D, K, K)
            B = pred_ebd.unsqueeze(2)  # (D, K, 1)
            B = B.expand(D, K, K)  # (D, K, K)
            expo = ((A - B)**2).mean(dim=0)  # (K, K)
            pred_similarity = 2 / (1 + torch.exp(expo))

            true_similarity = true_similarity.unsqueeze(0)
            pred_similarity = pred_similarity.unsqueeze(0)
            losses[i] = F.mse_loss(pred_similarity, true_similarity)
            # losses[i] = (true_similarity - pred_similarity).mean()

        return losses.mean()

    def extract(self, ebd, ann, size, device):
        size = torch.FloatTensor(size).to(device)
        kpts = torch.FloatTensor(ann['kpts']).to(device)
        mask_r = (0.0 <= kpts[:, 0]) & (kpts[:, 0] < 1.0)
        mask_c = (0.0 <= kpts[:, 1]) & (kpts[:, 1] < 1.0)
        mask = mask_r & mask_c

        true_ebd = torch.LongTensor(ann['tags']).to(device)
        true_ebd = true_ebd[mask]
        kpts = torch.floor(kpts[mask] * size).long()
        pred_ebd = ebd[:, kpts[:, 0], kpts[:, 1]]

        return true_ebd, pred_ebd


class PoseTagger:
    def __init__(self, log_dir, device):
        self.device = device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        self.model = Backbone().to(device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = TagLoss()

    def fit(self, train_set, valid_set, vis_set, epoch=100):
        self.train_loader = DataLoader(train_set, batch_size=25,
            shuffle=True, collate_fn=COCOKeypoint.collate_fn, num_workers=6)
        self.valid_loader = DataLoader(valid_set, batch_size=25,
            shuffle=False, collate_fn=COCOKeypoint.collate_fn, num_workers=4)
        self.vis_loader = DataLoader(vis_set, batch_size=25,
            shuffle=False, collate_fn=COCOKeypoint.collate_fn, num_workers=4)

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
        for img_batch, ann_batch in iter(self.train_loader):
            img_batch = img_batch.to(self.device)

            self.optim.zero_grad()
            ebd_batch = self.model(img_batch)
            loss = self.criterion(ebd_batch, ann_batch)
            loss.backward()
            self.optim.step()

            self.msg['loss'].update(loss.item())
            self.pbar.set_postfix(self.msg)
            self.pbar.update(len(img_batch))

    def _valid(self):
        self.msg.update({
            'val_loss': RunningAverage(),
        })
        self.model.eval()
        for img_batch, ann_batch in iter(self.valid_loader):
            img_batch = img_batch.to(self.device)
            ebd_batch = self.model(img_batch)
            loss = self.criterion(ebd_batch, ann_batch)
            self.msg['val_loss'].update(loss.item())
            self.pbar.update(len(img_batch))
        self.pbar.set_postfix(self.msg)

    def _vis(self):
        self.model.eval()
        idx = 0
        for img_batch, ann_batch in iter(self.vis_loader):
            ebd_batch = self.model(img_batch.to(self.device)).cpu()
            _, _, H, W = img_batch.size()
            ebd_batch = F.upsample(ebd_batch, (H, W))
            ebd_batch = ebd_batch * 0.45 + 0.5

            for img, ann, ebd in zip(img_batch, ann_batch, ebd_batch):
                ebd = pca(ebd, n_points=256, n_components=3)
                kpt = A.draw(img, ann)
                vis = img * 0.4 + ebd * 0.6
                path = f'{self.epoch_dir}/{idx:05d}.jpg'
                save_image([img, kpt, ebd, vis], path, nrow=4, pad_value=1)
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
        if self.ep == 0 or self.log['val_loss'].min() == self.msg['val_loss']:
            torch.save(self.model, str(self.epoch_dir / 'model.pth'))
