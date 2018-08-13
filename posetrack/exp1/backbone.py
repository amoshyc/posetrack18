from pathlib import Path
from datetime import datetime

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models import *

import matplotlib.pyplot as plt
plt.style.use('seaborn')
from tqdm import tqdm

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
            nn.Tanh(),
            nn.ConvTranspose2d(1024, 512, (2, 2), stride=2),
            nn.BatchNorm2d(512),
            nn.Tanh(),
            nn.ConvTranspose2d(512, 256, (2, 2), stride=2),
            nn.BatchNorm2d(256),
            nn.Tanh(),
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
        push = torch.zeros(batch_size, dtype=torch.float, device=device)
        pull = torch.zeros(batch_size, dtype=torch.float, device=device)

        for i in range(batch_size):
            ebds, ann = ebd_batch[i], ann_batch[i]
            vecs, tags = self.extract_vecs(ebds, ann)   # (ebd_dim, K)
            if vecs is None or tags is None:
                continue

            means = []
            for pid in range(ann['n_people']):
                person_mask = tags == pid
                if person_mask.sum() == 0:
                    continue
                person_vecs = vecs[:, person_mask]
                person_mean = person_vecs.mean(dim=1, keepdim=True)
                means.append(person_mean)
                pull[i] += torch.sqrt((person_vecs - person_mean)**2).mean()
            means = torch.stack(means, dim=0)
            means = means.squeeze(2)

            P = len(means)
            A = means.unsqueeze(0).expand(P, P, self.ebd_dim)
            B = means.unsqueeze(1).expand(P, P, self.ebd_dim)
            C = (-1/2) * (A - B)**2
            push[i] = torch.exp(C).mean()

        print(push)
        print(pull)
        return (push.mean() + pull.mean()) / 2

    def extract_vecs(self, ebds, ann):
        (_, H, W), device = ebds.size(), ebds.device
        if len(ann['kpts']) == 0:
            return None, None

        kpts = torch.FloatTensor(ann['kpts']).to(device)
        tags = torch.LongTensor(ann['tags']).to(device)
        idxs = torch.LongTensor(ann['idxs']).to(device)
        size = torch.FloatTensor([H, W]).to(device)

        mask_r = (0 <= kpts[:, 0]) & (kpts[:, 0] < 1)
        mask_c = (0 <= kpts[:, 1]) & (kpts[:, 1] < 1)
        mask = (mask_r & mask_c)
        if mask.sum() == 0:
            return None, None

        kpts = kpts[mask]
        tags = tags[mask]
        idxs = idxs[mask]
        kpts = torch.floor(kpts * size).long()

        all_vecs = []
        all_tags = []
        for i in range(17):
            ebd = ebds[i * self.ebd_dim:(i+1) * self.ebd_dim, :, :]
            part_mask = (idxs == i)
            if part_mask.sum() == 0:
                continue
            part_kpts = kpts[part_mask]
            all_vecs.append(ebd[:, part_kpts[:, 0], part_kpts[:, 1]])
            all_tags.append(tags[part_mask])
        all_vecs = torch.cat(all_vecs, dim=1)
        all_tags = torch.cat(all_tags, dim=0)
        return all_vecs, all_tags


class PoseEsitmator:
    def __init__(self, log_dir, device, model, optim, loss_fn):
        self.device = device
        self.log_dir = Path(log_dir).resolve()
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.model = model.to(device)
        self.optim = optim
        self.loss_fn = loss_fn
        self.log = []

    @staticmethod
    def from_scratch(log_dir, device):
        model = Backbone(3)
        optim = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = AELoss(3)
        return PoseEsitmator(log_dir, device, model, optim, loss_fn)

    def fit(self, train_set, valid_set, vis_set, epoch=30):
        self.train_loader = DataLoader(train_set, batch_size=25,
            shuffle=True, collate_fn=COCOKeypoint.collate_fn, num_workers=4)
        self.valid_loader = DataLoader(valid_set, batch_size=25,
            shuffle=False, collate_fn=COCOKeypoint.collate_fn, num_workers=4)
        self.vis_loader = DataLoader(vis_set, batch_size=25,
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
            'loss': util.RunningAverage(),
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
            'val_loss': util.RunningAverage(),
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

            img_batch = util.torch2np(img_batch)
            ebd_batch = util.torch2np(ebd_batch)

            for img, ann, ebd in zip(img_batch, ann_batch, ebd_batch):
                ebd = [ebd[:, :, i * 3:(i + 1) * 3] for i in range(17)]
                kpt = COCOKeypoint.draw_ann(img, ann)
                vis = [img * 0.4 + x * 0.6 for x in ebd]
                data = [img, *ebd, kpt, *vis]
                path = f'{self.epoch_dir}/{idx:05d}.jpg'
                save_grid(data, path, per_row=18, pad_value=1)
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
