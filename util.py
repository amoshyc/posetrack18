from math import ceil
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io
from skimage import color
from skimage import draw
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale


class RunningAverage:
    def __init__(self):
        self.iter = 0
        self.avg = 0.0

    def update(self, x):
        self.avg = (self.avg * self.iter + x) / (self.iter + 1)
        self.iter += 1

    def __str__(self):
        if self.iter == 0:
            return '-'
        return f'{self.avg:.4f}'


def pca(ebd, n_points=256, n_components=3):
    ebd = ebd.numpy().transpose([1, 2, 0])
    H, W, D = ebd.shape
    randp = np.random.rand(n_points, 2) * np.float32([H, W])
    randp = np.floor(randp).astype(np.int32)
    randp = ebd[randp[:, 0], randp[:, 1], :]

    pca = PCA(n_components=n_components).fit(randp)
    ebd = pca.transform(ebd.reshape(H * W, D))
    ebd = minmax_scale(ebd).reshape(H, W, -1)
    ebd = ebd.transpose([2, 0, 1])
    return torch.from_numpy(ebd)


def make_grid(arrs, per_row=-1, padding=2, pad_value=0):
    assert len(arrs) > 0
    for arr in arrs:
        assert arr.shape[:2] == arrs[0].shape[:2]

    arrs = copy(arrs)
    n_arr = len(arrs)
    for i in range(n_arr):
        if len(arrs[i].shape) == 2:
            arrs[i] = color.gray2rgb(arrs[i])
    for i in range(n_arr):
        if arrs[i].dtype == np.dtype(np.uint8):
            arrs[i] = skimage.img_as_float(arrs[i])

    imgH, imgW, _ = arrs[0].shape
    per_row = n_arr if per_row == -1 else per_row
    per_col = ceil(n_arr / per_row)
    gridW = per_row * imgW + (per_row - 1) * padding
    gridH = per_col * imgH + (per_col - 1) * padding
    grid = np.full((gridH, gridH, 3), pad_value, dtype=np.float64)
    for i in range(n_arr):
        c = (i % per_row) * (imgW + padding)
        r = (i // per_row) * (imgH + padding)
        grid[r:r+imgH, c:c+imgW] = arrs[i]

    return grid


def save_grid(arrs, filename, *args, **kwargs):
    grid = make_grid(arrs, *args, **kwargs)
    grid = (grid * 255).clip(0, 255).astype(np.uint8)
    io.imsave(filename, grid, quality=100)


def draw_ann(img, ann):
    kpts = np.int32(ann['kpts'])
    tags = np.int32(ann['tags'])

    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    colors = [c[:3] for c in colors]

    img = img.copy()
    for pid in range(ann['n_people']):
        person_kpts = kpts[tags == pid]
        for (r, c) in person_kpts:
            rr, cc = draw.circle(r, c, 2.1, shape=img.shape)
            img[rr, cc] = colors[pid % 20]

    return img
