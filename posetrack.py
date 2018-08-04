import json
from pathlib import Path
from pprint import pprint
from itertools import combinations

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F

ROOT_DIR = Path('/store/PoseTrack2017/posetrack_data/')

def convert(vann):
    result = []
    for fid, fann in enumerate(vann['annolist']):
        img_name = fann['image'][0]['name'].replace(r'\/', '/')
        w, h = Image.open(ROOT_DIR / img_name).size

        kpts, tags, idxs, boxs = [], [], [], []
        for pann in fann['annorect']:
            tag = pann['track_id'][0]
            boxs.append([
                pann['y1'][0] / h,
                pann['x1'][0] / w,
                (pann['y2'][0] - pann['y1'][0]) / h,
                (pann['x2'][0] - pann['x1'][0]) / w
            ])
            if not pann['annopoints']:
                continue
            for kpt in pann['annopoints'][0]['point']:
                x, y = kpt['x'][0], kpt['y'][0]
                if not kpt['is_visible'][0]:
                    continue
                kpts.append([y / h, x / w])
                idxs.append(kpt['id'][0])
                tags.append(tag)

        result.append({
            'image_name': img_name,
            'image_height': h,
            'image_width': w,
            'n_people': len(fann['annorect']),
            'kpts': kpts,
            'tags': tags,
            'idxs': idxs,
            'boxs': boxs,
        })
    return result


def pt_plot(img, anno, ax, origin_size=False, draw_outside=False):
    if len(anno['kpts']) == 0:
        print('No keypoints')
        ax.imshow(img)
        ax.axis('off')
        return

    colors = plt.cm.tab20(np.linspace(0.0, 1.0, 20))
    colors = [mpl.colors.to_hex(c) for c in colors]
    index = dict((x, i) for i, x in enumerate([
        'Rakl', 'Rkne', 'Rhip', 'Lhip', 'Lkne',
        'Lakl', 'Rwrt', 'Rebw', 'Rshd', 'Lshd',
        'Lebw', 'Lwrt', 'neck', 'nose', 'head'
    ]))
    edges = [
        ('head', 'nose'), ('nose', 'neck'),
        ('neck', 'Lshd'), ('Lshd', 'Lebw'), ('Lebw', 'Lwrt'),
        ('neck', 'Rshd'), ('Rshd', 'Rebw'), ('Rebw', 'Rwrt'),
        ('Lshd', 'Lhip'), ('Lhip', 'Rhip'), ('Rhip', 'Rshd'), ('Rshd', 'Lshd'),
        ('Lhip', 'Lkne'), ('Lkne', 'Lakl'), ('Rhip', 'Rkne'), ('Rkne', 'Rakl'),
    ]

    kpts = np.float32(anno['kpts'])
    tags = np.int32(anno['tags'])
    idxs = np.int32(anno['idxs'])

    if not draw_outside:
        mask_r = (0 <= kpts[:, 0]) & (kpts[:, 0] < 1.0)
        mask_c = (0 <= kpts[:, 1]) & (kpts[:, 1] < 1.0)
        mask = mask_r & mask_c
        kpts = kpts[mask, :]
        tags = tags[mask]
        idxs = idxs[mask]

    if origin_size:
        w, h = anno['image_width'], anno['image_height']
        img = img.resize((w, h))
    else:
        w, h = img.size
    kpts = kpts * np.float32([h, w])



    ax.axis('off')
    ax.imshow(img)
    for i in range(anno['n_people']):
        kpt = kpts[(tags == i)]
        ax.plot(kpt[:, 1], kpt[:, 0], '.', c=colors[i % 20])
        for (s, t) in edges:
            mask = (tags == i) & ((idxs == index[s]) | (idxs == index[t]))
            pos = kpts[mask]
            ax.plot(pos[:, 1], pos[:, 0], c=colors[i % 20])


class PoseTrack:
    def __init__(self, img_dir, ann_dir, img_size=(320, 320)):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.img_size = img_size
        json_paths = tqdm(list(self.ann_dir.glob('*.json')))
        self.pairs = []
        self.vanns = []
        for vid, vann_path in enumerate(json_paths):
            with vann_path.open() as f:
                vann = json.load(f)
            ids = [(vid, fid) for fid, fann in enumerate(vann) if len(fann['kpts']) > 0]
            for i in range(len(ids)):
                for j in range(i, len(ids)):  # min(i + 5, len(ids))
                    self.pairs.append((ids[i], ids[j]))
            self.vanns.append(vann)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        (vid1, fid1), (vid2, fid2) = self.pairs[idx]
        ann1 = self.vanns[vid1][fid1]
        ann2 = self.vanns[vid2][fid2]

        img1 = Image.open(self.img_dir / ann1['image_name']).convert('RGB')
        img2 = Image.open(self.img_dir / ann2['image_name']).convert('RGB')

        img1 = F.to_tensor(F.resize(img1, self.img_size))
        img2 = F.to_tensor(F.resize(img2, self.img_size))

        return img1, img2, ann1, ann2

    @staticmethod
    def collate_fn(batch):
        img1_batch = torch.stack([datum[0] for datum in batch], dim=0)
        img2_batch = torch.stack([datum[1] for datum in batch], dim=0)
        ann1_batch = list([datum[2] for datum in batch])
        ann2_batch = list([datum[3] for datum in batch])
        return img1_batch, img2_batch, ann1_batch, ann2_batch

    @staticmethod
    def plot_batch(img1_batch, img2_batch, ann1_batch, ann2_batch):
        pass



if __name__ == '__main__':
    # for mode in ['train', 'valid']:
    #     paths = sorted(list(Path(f'./gt17/{mode}/').glob('*.json')))
    #     dst_dir = Path(f'./pt17/{mode}/')
    #     dst_dir.mkdir(exist_ok=True, parents=True)
    #     for vid, path in enumerate(tqdm(paths)):
    #         with path.open() as f:
    #             vanno = json.load(f)
    #         anno = convert(vanno)
    #         with (dst_dir / f'{vid:04d}.json').open('w') as f:
    #             json.dump(anno, f, indent=2)

    # with open('./pt17/train/0000.json') as f:
    #     vann = json.load(f)
    # print('#Frames:', len(vann))
    # print('#Annotated:', [i for i, a in enumerate(vann) if len(a['kpts']) > 0])
    # anno = vann[30]
    # img = Image.open(ROOT_DIR / anno['image_name'])
    # fig, ax = plt.subplots(dpi=100)
    # pt_plot(img, anno, ax)
    # plt.show()

    ds = PoseTrack(ROOT_DIR, './pt17/train/')
    print(len(ds))
    img1, img2, ann1, ann2 = ds[2000]
    fig, ax = plt.subplots(1, 2, sharey=True, dpi=100, figsize=(16, 8))
    pt_plot(F.to_pil_image(img1), ann1, ax[0])
    pt_plot(F.to_pil_image(img2), ann2, ax[1])
    fig.tight_layout()
    plt.show()
