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

shape = imgs[1].shape
width, height = shape
count = len(imgs)
print("Load gifs count:{} shape:{}".format(count, shape))


flat_imgs = np.array([i.flatten() for i in imgs])
X = flat_imgs
print("X.shape:{}".format(X.shape))

Image.fromarray(np.reshape(X[30], (width, height))).show()

### hyper
d = 0.98

# pca = PCA(n_components=d, svd_solver="randomized")
pca = PCA(n_components=d, svd_solver="full")
Z = pca.fit_transform(flat_imgs)
print("Pca成分的方差:{}".format(pca.explained_variance_))
print("Pca成分的方差比例:{}".format(pca.explained_variance_ratio_))
W = pca.components_
# W = pca.singular_values_
print("Z.shape:{} W.shape:{}".format(Z.shape, W.shape))
### 利用 XX = ZW' 还原X
XX = np.dot(Z, W)
XX = np.reshape(XX, (count, width, height))

Image.fromarray(XX[30]).show()

