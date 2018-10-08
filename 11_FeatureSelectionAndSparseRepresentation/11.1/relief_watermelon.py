# coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import os

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR+"/../Model")
sys.path.append(ROOT_DIR+"/../../0_FeatureMaker")
import numpy as np
from feature_maker import FeatureMaker
import json
import heapq
from collections import Counter

def relief_diff(x, y, feature_types):
  diff = []
  for j,t in enumerate(feature_types):
    if t==0:
      diff_j = int(x[j]!=y[j])
    else:
      diff_j = x[j]-y[j]
    diff.append(diff_j)
  return np.array(diff)

def nearest_index(l, y):
  nearest = [float('inf') for c in set(y)]
  indice = [0 for c in nearest]
  for i,dist in enumerate(l):
    c = int(y[i])
    if dist<nearest[c]:
      nearest[c] = dist
      indice[c] = i
  return indice


src = ROOT_DIR+'/../../Data/watermelon/watermelon_3.0.csv'
feature_types = [0,0,0,0,0,0,1,1,0]
feature_maker = FeatureMaker(src=src, delimiter=',', types=feature_types, norm=True)
X, Y = feature_maker.make(skip_rows=1, skip_cols=1, one_hot=False)
Y = Y.flatten()
print("X:{}\nY:{}".format(X, Y))

### hyper
k = 4

m, n = X.shape
delta = np.zeros((X.shape[1],))
### 样本距离矩阵
dists = np.zeros((m,m))
# print(dists)
for i in range(m):
  for j in range(i,m):
    if i==j:
      dists[i][j] = float('inf')
      continue
    dist = np.linalg.norm(relief_diff(X[i], X[j], feature_types=feature_types[:-1]))
    # print(dist)
    dists[i][j] = dist
    dists[j][i] = dist
# print("Dist:\n{}".format(dists))

delta = 0
for i,y in enumerate(Y):
  x = X[i]
  c = int(y)
  nearest = nearest_index(dists[i], Y)
  # print(nearest)
  homo_diff = relief_diff(x, X[nearest[c]], feature_types=feature_types[:-1])
  hetero_diff = relief_diff(x, X[nearest[(c+1)%2]], feature_types=feature_types[:-1])
  delta += -homo_diff**2 + hetero_diff**2
  # break

print("delta:{}".format(delta))

sorted_fea_index = sorted(range(n), key=delta.take, reverse=True)
print("feature chosen in order:{}".format(sorted_fea_index))


