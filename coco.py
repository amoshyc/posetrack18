import json
from random import randint
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.utils import save_image

from util import Annotation as A

COCO_INDEX = dict((x, i) for i, x in enumerate([
    'nose', 'Leye', 'Reye', 'Lear', 'Rear', 'Lshd', 'Rshd', 'Lebw', 'Rebw',
    'Lwrt', 'Rwrt', 'Lhip', 'Rhip', 'Lkne', 'Rkne', 'Lakl', 'Rakl'
]))
COCO_EDGES = [
    # face
    ('Lear', 'Leye'), ('Leye', 'nose'),
    ('Leye', 'Reye'), ('Reye', 'nose'), ('Reye', 'Rear'),
    # upper body
    ('Leye', 'Lshd'), ('Lshd', 'Lebw'), ('Lebw', 'Lwrt'),
    ('Reye', 'Rshd'), ('Rshd', 'Rebw'), ('Rebw', 'Rwrt'),
    # torso
    ('Lshd', 'Lhip'), ('Lhip', 'Rhip'), ('Rhip', 'Rshd'), ('Rshd', 'Lshd'),
    # lower body
    ('Lhip', 'Lkne'), ('Lkne', 'Lakl'), ('Rhip', 'Rkne'), ('Rkne', 'Rakl'),
]


def parse_coco(src_path, dst_path):
    def extract(annos, h, w):
        kpts, tags, idxs, boxs = [], [], [], []
        for tag, anno in enumerate(annos):
            kpt = np.float32(anno['keypoints']).reshape(-1, 3)
            tag = np.ones(len(kpt)) * tag
            idx = np.tile(np.arange(N_PARTS), len(kpt) // N_PARTS)
            box = np.float32(anno['bbox'])
            vis = kpt[:, 2]
            kpt = kpt[:, [1, 0]] # xy -> rc
            box = box[[1, 0, 3, 2]] # xywh -> crhw
            kpt = kpt / np.float32([h, w])          # normalized
            box = box / np.float32([h, w, h, w])    # normalized
            mask = (vis > 0)  # 0: invisible, 1: occulation, 2: visible
            kpts.extend(kpt[mask].tolist())
            tags.extend(tag[mask].astype(np.int32).tolist())
            idxs.extend(idx[mask].astype(np.int32).tolist())
            boxs.append(box.tolist())
        return kpts, tags, idxs, boxs

    N_PARTS = 17
    src_path = Path(src_path).resolve()
    dst_path = Path(dst_path).resolve()
    dst_path.parent.mkdir(exist_ok=True)
    with src_path.open() as f:
        data = json.load(f)

    images = dict()
    for anno in data['images']:
        images[anno['id']] = anno

    annos_by_image = defaultdict(list)
    for anno in data['annotations']:
        if anno['num_keypoints'] > 0:
            annos_by_image[anno['image_id']].append(anno)

    result = []
    for image_id, annos in tqdm(annos_by_image.items()):
        h, w = images[image_id]['height'], images[image_id]['width']
        kpts, tags, idxs, boxs = extract(annos, h, w)
        result.append({
            'image_name': images[image_id]['file_name'],
            'image_height': h,
            'image_width': w,
            'n_people': len(annos), # Range Z+
            'kpts': kpts,  # positions, shape: (K, 2), range: [0.0, 1.0]
            'tags': tags,  # which person, shape: (K,), range: Z+
            'idxs': idxs,  # which keypoint, shape: (K,), range: [0, 17)
            'boxs': boxs,  # bounding boxs, shape: (K, 4), range: [0.0, 1.0]
        })

    with dst_path.open('w') as f:
        json.dump(result, f, indent=2)


def coco_plot(img, anno, ax):
    w, h = img.size
    kpts = np.float32(anno['kpts']) * np.float32([h, w])
    tags = np.int32(anno['tags'])
    idxs = np.int32(anno['idxs'])
    boxs = np.float32(anno['boxs']) * np.float32([h, w, h, w])

    colors = plt.cm.tab20(np.linspace(0.0, 1.0, 20))
    colors = [mpl.colors.to_hex(c) for c in colors]

    ax.axis('off')
    ax.imshow(img)
    for i in range(anno['n_people']):
        kpt = kpts[(tags == i)]
        ax.plot(kpt[:, 1], kpt[:, 0], '.', c=colors[i % 20])
        for (s, t) in COCO_EDGES:
            mask = (tags == i) & ((idxs == COCO_INDEX[s]) | (idxs == COCO_INDEX[t]))
            pos = kpts[mask]
            ax.plot(pos[:, 1], pos[:, 0], c=colors[i % 20])
        # y, x, h, w = boxs[i]
        # rect = mpl.patches.Rectangle((x, y), w, h, fill=False, ec=colors[i % 20], lw=1)
        # ax.add_patch(rect)


class COCOKeypoint:
    def __init__(self, img_dir, ann_path, img_size):
        self.img_dir = Path(img_dir)
        self.ann_path = Path(ann_path)
        self.img_size = img_size
        with self.ann_path.open() as f:
            data = json.load(f)
        self.anns = [ann for ann in data if ann['n_people'] >= 2]
        self.extend_edges = [(COCO_INDEX[s], COCO_INDEX[t]) for (s, t) in [
            ('Lshd', 'Rhip'), ('Rshd', 'Lhip'), ('Rshd', 'Lshd'),
            ('Lshd', 'Lebw'), ('Lebw', 'Lwrt'), ('Rshd', 'Rebw'), ('Rebw', 'Rwrt'),
            ('Lhip', 'Lkne'), ('Lkne', 'Lakl'), ('Rhip', 'Rkne'), ('Rkne', 'Rakl')
        ]]

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann = self.anns[idx]
        img = Image.open(self.img_dir / ann['image_name']).convert('RGB')
        img = F.to_tensor(F.resize(img, self.img_size))
        ann = A.extend(ann, self.extend_edges, 4)
        return img, ann

    def transform(self, img, ann):
        if randint(0, 1) == 0:
            img = F.hflip(img)
            ann = A.hflip(img)

        if randint(0, 3) == 0:
            img = F.to_grayscale(img)
        else:
            img = T.ColorJitter(0.3, 0.3, 0.2)(img)


    @staticmethod
    def collate_fn(batch):
        img_batch = torch.stack([item[0] for item in batch], dim=0)
        ann_batch = list([item[1] for item in batch])
        return img_batch, ann_batch


if __name__ == '__main__':
    # src_path = Path(f'/store/COCO/annotations/person_keypoints_train2017.json')
    # dst_path = Path(f'./coco/train.json')
    # parse_coco(src_path, dst_path)
    # src_path = Path(f'/store/COCO/annotations/person_keypoints_val2017.json')
    # dst_path = Path(f'./coco/valid.json')
    # parse_coco(src_path, dst_path)

    img_dir = '/store/COCO/val2017/'
    ann_path = './coco/valid.json'
    ds = COCOKeypoint(img_dir, ann_path, (256, 256))
    print(len(ds))
    # dl = DataLoader(ds, batch_size=2, collate_fn=COCOKeypoint.collate_fn)
    # img_batch, ann_batch = next(iter(dl))
    # print(img_batch.size())
    # print(len(ann_batch))

    for i in [0, 1, 2, 3]:
        img, ann = ds[i]
        img = A.draw(img, ann)
        save_image([img], f'./test{i}.jpg')

    # img, ann = ds[6]
    # img = F.to_pil_image(img)
    # fig, ax = plt.subplots(dpi=100)
    # coco_plot(img, ann, ax)
    # plt.show()
