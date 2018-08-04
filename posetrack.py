import json
from pathlib import Path
from pprint import pprint

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn')

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
                if kpt['is_visible'][0]:
                    kpts.append([kpt['y'][0] / h, kpt['x'][0] / w])
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


def pt_plot(img, anno, ax):
    if len(anno['kpts']) == 0:
        print('No keypoints')
        ax.imshow(img)
        ax.axis('off')
        return

    w, h = img.size
    kpts = np.float32(anno['kpts']) * np.float32([h, w])
    boxs = np.float32(anno['boxs']) * np.float32([h, w, h, w])
    tags = np.int32(anno['tags'])
    idxs = np.int32(anno['idxs'])

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

    with open('./pt17/train/0000.json') as f:
        vann = json.load(f)

    print('#Frames:', len(vann))
    print('#Annotated:', [i for i, a in enumerate(vann) if len(a['kpts']) > 0])

    anno = vann[30]
    img = Image.open(ROOT_DIR / anno['image_name'])
    fig, ax = plt.subplots(dpi=100)
    pt_plot(img, anno, ax)
    plt.show()
