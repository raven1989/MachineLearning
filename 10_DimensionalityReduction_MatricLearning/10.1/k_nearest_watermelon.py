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

src = ROOT_DIR+'/../../Data/watermelon/watermelon_3.0_x.csv'
feature_types = [1,1,0]
feature_maker = FeatureMaker(src=src, delimiter=',', types=feature_types, norm=False)
X, Y = feature_maker.make(skip_rows=1, skip_cols=1, one_hot=False)
Y = Y.flatten()
print("X:{}\nY:{}".format(X, Y))

### hyper
k = 3

test_X = X
predict = []

for x in test_X:
  dist = np.linalg.norm(x-X, axis=1)
  k_nearest = heapq.nsmallest(k, range(len(dist)), key=dist.take)
  pre = Counter(Y[k_nearest]).most_common(1)[0][0]
  predict.append(pre)
  # print("dist:{}".format(dist))
  print("{} nearest:{} pre:{}".format(k, k_nearest, pre))

predict = np.array(predict)

accuracy = np.mean(Y==predict)

print("Y:      {}\nPredict:{}\nAccuracy:{}".format(Y, predict, accuracy))

