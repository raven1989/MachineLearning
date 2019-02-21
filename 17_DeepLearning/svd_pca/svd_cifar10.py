# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import os

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
import glob
import numpy as np
from PIL import Image
import cPickle
# from sklearn.decomposition import PCA

src = ROOT_DIR+'/../../Data/cifar10/test_batch'
meta = ROOT_DIR+'/../../Data/cifar10/batches.meta'

raw_dict = {}
with open(src, 'rb') as fo:
  raw_dict = cPickle.load(fo)

Xraw = raw_dict[b'data']
Y = raw_dict[b'labels']

label_names = []
with open(meta, 'rb') as fo:
  label_names = cPickle.load(fo)[b'label_names']

def trans_cifar10_to_img(x):
  height, width, channel = 32, 32, 3
  img = np.reshape(x, (channel,height,width)).transpose((1,2,0))
  return Image.fromarray(img, 'RGB')

def tile_up_img(imgs, img_shape, shape):
  width, height = img_shape
  m, n = shape
  max_width, max_height = width*m, height*n
  i = 0
  canvas = Image.new('RGB', (max_width, max_height), 255)
  for y in range(0, max_height, height):
    for x in range(0, max_width, width):
      canvas.paste(imgs[i], (x,y))
      i += 1
  return canvas

  
### 这里的样本越多，奇异值分解后获得的奇异向量越好，降维d就可以选择地越小，但计算协方差矩阵量会很大
### 转成浮点型，numpy可以并行计算dot，整型就是单核很慢
X = Xraw[:].astype(np.float32)

# ### 按列中心化，每一列看做一维
mu = np.mean(X, axis=0)
# print(mu)
X -= mu

print("X.shape:{}".format(X.shape))

### 计算协方差矩阵
cov = np.dot(X.T, X)/X.shape[0]
# print(cov.shape)

### 奇异值分解
U, S, V = np.linalg.svd(cov)
# print(U.shape)
# print(V.shape)


### 降维
### 如果d取D=3*32*32=3072，则就没有降维，只是将图像转换到新的坐标系，然后再转换回来，图像理应不变，可以作为验证
# d = 3072
d = 144

### 可视化奇异向量
m, n = int(np.sqrt(d)), int(np.sqrt(d))
tile_cnt = d
e = U[:,:tile_cnt]
### 把奇异向量规范到0-255
e = (e-e.min(axis=0))/(e.max(axis=0)-e.min(axis=0))*255
e = e.astype(np.uint8)
# print(e)
tiles = [trans_cifar10_to_img(e[:,j]) for j in range(tile_cnt)]
tile_up_img(tiles, img_shape=(32,32), shape=(m,n)).show()

### decorelate data 相当于坐标系变换，变换到求出来的特征向量作为基的坐标系上
# Xrot = np.dot(X, U)
# Xrot_reduced = Xrot[:,:d]

### 上面这两步合起来相当于
Xrot_reduced = np.dot(X, U[:,:d])

### 把降维后的数据再转换回原坐标系
Xret = np.dot(Xrot_reduced, U.transpose()[:d,:])

### 把恢复后的数据转成rgb类型并去中心化到0-255范围
Xret = (Xret+mu).astype(np.uint8)

# print(Xraw)
# print(Xret)

if len(sys.argv)>1:
  check_root = int(sys.argv[1])
  check_cnt = check_root**2
  print("Y[{}]:{}".format(check_cnt, label_names[Y[check_cnt]]))
  # trans_cifar10_to_img(Xraw[check_idx]).show()
  # trans_cifar10_to_img(Xret[check_idx]).show()
  origin_imgs = [trans_cifar10_to_img(Xraw[i]) for i in range(check_cnt)]
  reduce_imgs = [trans_cifar10_to_img(Xret[i]) for i in range(check_cnt)]
  tile_up_img(origin_imgs, img_shape=(32,32), shape=(check_root,check_root)).show()
  tile_up_img(reduce_imgs, img_shape=(32,32), shape=(check_root,check_root)).show()


