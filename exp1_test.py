from skimage import io
from posetrack import util
from posetrack.exp1.coco import COCOKeypoint

img_dir = '/store/COCO/val2017/'
ann_path = './data/coco/valid.json'
ds = COCOKeypoint(img_dir, ann_path, (256, 256), mode='train')
print(len(ds))

img, ann = ds[2]
vis = COCOKeypoint.draw_ann(img, ann)
util.save_grid([vis], 'test.png')
# io.imshow(vis)
# io.show()
