from random import randint
from datetime import datetime

import matplotlib as mpl
mpl.use('Agg')

import torch
from torch.utils.data.dataset import Subset, ConcatDataset

from posetrack.exp2.coco import COCOKeypoint
from posetrack.exp2.backbone import PoseTagger

train_img_dir = '/store/COCO/train2017/'
valid_img_dir = '/store/COCO/val2017/'
COCOtrain = COCOKeypoint(
    train_img_dir, './data/coco/train.json', (320, 320), mode='train')
COCOvalid = COCOKeypoint(
    valid_img_dir, './data/coco/valid.json', (320, 320), mode='eval')
train_vis = [randint(0, len(COCOtrain) - 1) for _ in range(50)]
valid_vis = [randint(0, len(COCOvalid) - 1) for _ in range(50)]
COCOvis = ConcatDataset(
    [Subset(COCOtrain, train_vis),
     Subset(COCOvalid, valid_vis)])
print('#Train:', len(COCOtrain))
print('#Valid:', len(COCOvalid))
print('Train Vis:', train_vis)
print('Valid Vis:', valid_vis)

log_dir = f'./log/{datetime.now():%m-%d %H:%M:%S}/'
device = torch.device('cuda')
tagger = PoseTagger(log_dir, device)
tagger.fit(COCOtrain, COCOvalid, COCOvis, 30)
