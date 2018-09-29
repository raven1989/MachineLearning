# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import os

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
import glob
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

src = ROOT_DIR+'/../../Data/yalefaces/subject*'

imgs = []
for gif in glob.iglob(src):
  im = Image.open(gif)
  imgs.append(np.asarray(im))
# im.show()
shape = imgs[0].shape
print("Load gifs count:{} shape:{}".format(len(imgs), shape))

flat_imgs = np.array([i.flatten() for i in imgs])
print("X.shape:{}".format(flat_imgs.shape))

### hyper
d = 20

pca = PCA(n_components=d)
res = pca.fit_transform(flat_imgs)
print("Result.shape:{}".format(res.shape))

# Image.fromarray(np.reshape(res[0],(50,40))).show()

# res = Image.fromarray(img)
# res.show()
