from pathlib import Path
from datetime import datetime

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models import *

import matplotlib.pylot as plt
plt.style.use('seaborn')

from .coco import COCOKeypoint
from .. import util

class Backbone(nn.Module):
    def __init__(self, ebd_dim=3):
        super().__init__()

        resnet = resnet50(pretrained=True)
        self.pre = nn.Sequential(
            nn.BatchNorm2d(3),
        )
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, (2, 2), stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, (2, 2), stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, (2, 2), stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.post = nn.Sequential(
            nn.Conv2d(256, 17 * ebd_dim, (1, 1)),
            nn.BatchNorm2d(17 * ebd_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.pre(x)
        x = self.features(x)
        x = self.deconv(x)
        x = self.post(x)
        return x


class AELoss(nn.Module):
    def __init__(self, ebd_dim):
        super().__init__()
        self.ebd_dim = ebd_dim

    def forward(self, ebd_batch, ann_batch):
        batch_size, device = ebd_batch.size(0), ebd_batch.device
        push = torch.zeros(B, dtype=torch.float, device=device)
        pull = torch.zeros(B, dtype=torch.float, device=device)

        for i in range(batch_size):
            ebds, ann = ebd_batch[i], ann_batch[i]
            P = ann['n_people']
            vecs, tags = self.extract_vecs(ebds, ann)   # (ebd_dim, K)

            means = []
            for pid in range(P):
                person_vecs = vecs[tags == pid]
                person_mean = person_vecs.mean(dim=1, keepdim=True)
                means.append(person_mean)
                pull[i] += torch.sqrt((person_vecs - person_mean)**2).mean()
            means = torch.cat(means, dim=0).squeeze()   # (P, ebd_dim)

            A = means.unsqueeze(0).expand(P, P, self.ebd_dim)
            B = means.unsqueeze(1).expand(P, P, self.ebd_dim)
            push[i] = torch.sqrt((A - B)**2).mean()

        return (push.mean() + pull.mean()) / 2

    def extract_vecs(self, ebds, ann):
        (_, H, W), device = ebds.size(), ebds.device
        kpts = ann['kpts'] * torch.FloatTensor([H, W]).to(device)
        kpts = torch.floor(kpts).long()

        vecs = []
        tags = []
        for i in range(17):
            ebd = ebds[i * self.ebd_dim:(i+1) * self.ebd_dim]
            mask = ann['idxs'] == i
            part_kpts = kpts[mask]
            vecs.append(ebd[:, part_kpts[:, 0], part_kpts[:, 1]])
            tags.append(ann['tags'][mask])
        vecs = torch.cat(vecs, dim=0)
        tags = torch.cat(tags, dim=0)
        return vecs, tags


class PoseEsitmator:
    def __init__(self, log_dir, device, model, optim, loss_fn):
        self.device = device
        self.log_dir = Path(log_dir).resolve()
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.model = model
        self.optim = optim
        self.loss_fn = loss_fn

    @staticmethod
    def from_scratch(log_dir, device):
        model = Backbone()
        optim = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = AELoss()
        return PoseEsitmator(log_dir, device, model, optim, loss_fn)

    def fit(self, train_set, valid_set, vis_set, epoch=30):
        self.train_loader = DataLoader(train_set, batch_size=32,
            shuffle=True, collate_fn=COCOKeypoint.collate_fn, num_workers=4)
        self.valid_loader = DataLoader(valid_set, batch_size=32,
            shuffle=False, collate_fn=COCOKeypoint.collate_fn, num_workers=4)
        self.vis_loader = DataLoader(vis_set, batch_size=32,
            shuffle=False, collate_fn=COCOKeypoint.collate_fn, num_workers=4)

        for self.ep in range(epoch):
            self.log.append({
                'epoch': self.ep
            })
            tqdm_args = {
                'total': len(train_set) + len(valid_set),
                'desc': f'Epoch {self.ep:03d}',
                'ascii': True
            }
            with tqdm(**tqdm_args) as self.pbar:
                self._train()
                with torch.no_grad():
                    self._valid()
                    self._vis()
                    self._log()

    def _train(self):
        self.log[-1].update({
            'loss': RunningAverage(),
        })
        self.model.train()
        for img_batch, ann_batch in iter(self.train_loader):
            img_batch = img_batch.to(self.device)

            self.optim.zero_grad()
            ebd_batch = self.model(img_batch)
            loss = self.loss_fn(ebd_batch, ann_batch)
            loss.backward()
            self.optim.step()

            self.log[-1]['loss'].update(loss.item())
            self.pbar.set_postfix(self.log[-1])
            self.pbar.update(len(img_batch))

    def _valid(self):
        self.log[-1].update({
            'val_loss': RunningAverage(),
        })
        self.model.eval()
        for img_batch, ann_batch in iter(self.valid_loader):
            img_batch = img_batch.to(self.device)
            ebd_batch = self.model(img_batch)
            loss = self.loss_fn(ebd_batch, ann_batch)
            self.log[-1]['val_loss'].update(loss.item())
            self.pbar.update(len(img_batch))
        self.pbar.set_postfix(self.log[-1])

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
        df = pd.DataFrame(self.log)
        df.to_csv(str(self.log_dir / 'log.csv'))
        # plot loss
        fig, ax = plt.subplots(dpi=100)
        df[['loss', 'val_loss']].plot(ax=ax)
        fig.tight_layout()
        fig.savefig(str(self.log_dir / 'loss.jpg'))
        plt.close()  # Close plot to prevent RE
        # model
        if self.log['val_loss'].idxmin() == self.ep:
            torch.save(self.model, str(self.epoch_dir / 'model.pth'))
