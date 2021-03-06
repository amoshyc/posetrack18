import json
from random import randint
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import skimage
from skimage import io
from skimage import draw
from skimage import color
from skimage import transform

import imgaug as ia
from imgaug import augmenters as iaa
ia.seed(randint(0, 100))

from .. import util


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


class COCOKeypoint:
    def __init__(self, img_dir, ann_path, img_size, mode='eval'):
        self.img_dir = Path(img_dir)
        self.ann_path = Path(ann_path)
        self.img_size = img_size
        self.mode = mode
        with self.ann_path.open() as f:
            self.anns = json.load(f)

        self.train_aug = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=10, scale=(1.0, 1.5)),
            iaa.Scale({
                "height": self.img_size[0],
                "width": self.img_size[1]
            }),
            iaa.AdditiveGaussianNoise(
                loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            iaa.Add((-10, 10), per_channel=0.5),
            iaa.AddToHueAndSaturation((-20, 20)),
        ])
        self.eval_aug = iaa.Sequential([
            iaa.Scale({
                "height": self.img_size[0],
                "width": self.img_size[1]
            })
        ])

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann = self.anns[idx]
        img_path = self.img_dir / ann['image_name']
        img = io.imread(str(img_path))
        if img.ndim == 2:
            img = color.gray2rgb(img)

        if self.mode == 'train':
            img, ann = self.train_transform(img, ann)
        if self.mode == 'eval':
            img, ann = self.eval_transform(img, ann)
        img = skimage.img_as_float(img)

        return img, ann

    def train_transform(self, img, ann):
        aug = self.train_aug.to_deterministic()

        kpts = np.float32(ann['kpts'])
        tags = np.int32(ann['tags'])
        idxs = np.int32(ann['idxs'])
        imgH = ann['image_height']
        imgW = ann['image_width']

        aug_img = aug.augment_images([img])[0]
        kpts = np.floor(kpts * np.float32([imgH, imgW])).astype(np.int32)
        src_kpts = ia.KeypointsOnImage.from_coords_array(kpts[:, ::-1], shape=img.shape)
        aug_kpts = aug.augment_keypoints([src_kpts])[0].get_coords_array()[:, ::-1]
        aug_kpts = aug_kpts / np.float32(self.img_size)

        mask_r = (0 <= aug_kpts[:, 0]) & (aug_kpts[:, 0] < self.img_size[0])
        mask_c = (0 <= aug_kpts[:, 1]) & (aug_kpts[:, 1] < self.img_size[1])
        mask = mask_r & mask_c

        aug_ann = ann.copy()
        aug_ann['kpts'] = aug_kpts[mask].tolist()
        aug_ann['tags'] = tags[mask].tolist()
        aug_ann['idxs'] = idxs[mask].tolist()
        return aug_img, aug_ann

    def eval_transform(self, img, ann):
        img = transform.resize(img, self.img_size, mode='reflect', anti_aliasing=True)
        return img, ann

    @staticmethod
    def collate_fn(batch):
        img_batch = np.stack([item[0] for item in batch], axis=0)
        img_batch = util.np2torch(img_batch).float()
        ann_batch = list([item[1] for item in batch])
        return img_batch, ann_batch

    @staticmethod
    def draw_ann(img, ann):
        img = img * 0.5
        if len(ann['kpts']) == 0:
            return img

        H, W, _ = img.shape
        kpts = np.float32(ann['kpts']) * np.float32([H, W])
        kpts = np.floor(kpts).astype(np.int32)
        tags = np.int32(ann['tags'])

        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        colors = [c[:3] for c in colors]

        for pid in range(ann['n_people']):
            person_kpts = kpts[tags == pid]
            for (r, c) in person_kpts:
                rr, cc = draw.circle(r, c, 2.3, shape=img.shape)
                img[rr, cc] = colors[pid % 20]

        return img
