from random import randint
from datetime import datetime

import matplotlib as mpl
mpl.use('Agg')
import torch
from torch.utils.data.dataset import Subset, ConcatDataset

from coco import COCOKeypoint
from backbone import PoseTagger

train_img_dir = '/store/COCO/train2017/'
valid_img_dir = '/store/COCO/val2017/'
COCOtrain = COCOKeypoint(train_img_dir, './coco/train.json', (320, 320), mode='train')
COCOvalid = COCOKeypoint(valid_img_dir, './coco/valid.json', (320, 320), mode='eval')
train_vis = [randint(0, len(COCOtrain) - 1) for _ in range(50)]
valid_vis = [randint(0, len(COCOvalid) - 1) for _ in range(50)]
COCOvis = ConcatDataset([
    Subset(COCOtrain, train_vis),
    Subset(COCOvalid, valid_vis)
])
print('#Train:', len(COCOtrain))
print('#Valid:', len(COCOvalid))
print('Train Vis:', train_vis)
print('Valid Vis:', valid_vis)

log_dir = f'./log/{datetime.now():%m-%d %H:%M:%S}/'
device = torch.device('cuda')
tagger = PoseTagger(log_dir, device)
tagger.fit(COCOtrain, COCOvalid, COCOvis, 30)

# from posetrack import PoseTrack
# from tracker import PoseTracker

# root_dir = '/store/PoseTrack2017/posetrack_data/'

# PTtrain = PoseTrack(root_dir, './pt17/train/', (256, 256))
# PTvalid = PoseTrack(root_dir, './pt17/valid/', (256, 256))
# train_vis = [randint(0, len(PTtrain) - 1) for _ in range(50)]
# valid_vis = [randint(0, len(PTvalid) - 1) for _ in range(50)]
# PTvis = ConcatDataset([
#     Subset(PTtrain, train_vis),
#     Subset(PTvalid, valid_vis)
# ])
# print('#Train:', len(PTtrain))
# print('#Valid:', len(PTvalid))
# print('Train Vis:', train_vis)
# print('Valid Vis:', valid_vis)

# log_dir = f'./log/{datetime.now():%m-%d %H:%M:%S}/'
# device = torch.device('cuda')
# backbone = torch.load('./log/08-07 20:45:37/010/model.pth')
# tracker = PoseTracker(log_dir, device, backbone)
# tracker.fit(PTtrain, PTvalid, PTvis, epoch=50)
