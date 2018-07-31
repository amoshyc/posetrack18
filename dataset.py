import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from tqdm import tqdm

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
    index = dict((x, i) for i, x in enumerate([
        'nose', 'Leye', 'Reye', 'Lear', 'Rear', 'Lshd', 'Rshd', 'Lebw', 'Rebw',
        'Lwrt', 'Rwrt', 'Lhip', 'Rhip', 'Lkne', 'Rkne', 'Lakl', 'Rakl'
    ]))
    edges = [
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

    ax.axis('off')
    ax.imshow(img)
    for i in range(anno['n_people']):
        kpt = kpts[(tags == i)]
        ax.plot(kpt[:, 1], kpt[:, 0], '.', c=colors[i % 20])
        for (s, t) in edges:
            mask = (tags == i) & ((idxs == index[s]) | (idxs == index[t]))
            pos = kpts[mask]
            ax.plot(pos[:, 1], pos[:, 0], c=colors[i % 20])
        y, x, h, w = boxs[i]
        rect = mpl.patches.Rectangle((x, y), w, h, fill=False, ec=colors[i % 20], lw=1)
        ax.add_patch(rect)


class COCOSiameseTag:
    def __init__(self, image_dir, json_path, img_size, hmp_size, mode='train'):
        self.image_dir = Path(image_dir)
        self.json_path = Path(json_path)
        self.img_size = img_size
        self.hmp_size = hmp_size

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def _make_pair(self, img, kpt):
        pass


if __name__ == '__main__':
    # for mode in ['train', 'val']:
    #     src_path = Path(f'/store/COCO/annotations/person_keypoints_{mode}2017.json')
    #     dst_path = Path(f'./data/{mode}2017.json')
    #     parse_coco(src_path, dst_path)

    with open('./data/val2017.json') as f:
        data = json.load(f)
    for anno in data:
        if anno['n_people'] > 4:
            break
    img_path = f'/store/COCO/val2017/{anno["image_name"]}'
    img = Image.open(img_path)
    fig, ax = plt.subplots(dpi=100)
    coco_plot(img, anno, ax)
    plt.show()
