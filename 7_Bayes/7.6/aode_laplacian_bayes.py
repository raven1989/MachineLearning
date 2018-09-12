#! coding:utf-8

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
import bisect

src = ROOT_DIR+'/../../Data/watermelon/watermelon_3.0.csv'
feature_types = [0,0,0,0,0,0,1,1,0]
feature_maker = FeatureMaker(src=src, delimiter=',', types=feature_types, norm=False)
X, Y = feature_maker.make(skip_rows=1, skip_cols=1, one_hot=True)
Y = Y.flatten()
print("X:{}\nY:{}".format(X, Y))

### train
num_sample = X.shape[0]
num_continuous_feature = np.sum(feature_types[:-1])
discrete_X = X[:,:-num_continuous_feature]
continuous_X = X[:,num_continuous_feature:]

### label种类数
num_class = len(feature_maker.feature2index[-1])

### 离散特征
### 特征的属性数量总和
num_property = discrete_X.shape[1]
print("sample num:{} class_num:{} ".format(num_sample, num_class))
### 属性矩阵 num_class x num_property x num_property
### 每一个元素N(c,i,j) 表示 (第c类 & 特征fa取值为i & 特征fb取值为j)
N_shape = (num_class, num_property, num_property)
print("N.shape:{}".format(N_shape))
N = np.zeros(N_shape)
### 特征分组 [i:i+1] 表示一个feature分组，由其property铺平而成
feature_partition = feature_maker.one_hot_encoder.feature_indices_
print("feature partition:{}".format(feature_partition))
for i,x in enumerate(discrete_X):
  x = np.reshape(x, (x.shape[0],1))
  c = int(Y[i])
  # print("c:{} x:{}".format(c, np.dot(x, x.T)))
  N[c] += np.dot(x, x.T)
  ### 将对角线置为0，因为对角线是属性自己计数，只要属性i存在则，N(c,i,i)必为1
  # np.fill_diagonal(N[c], np.zeros(num_property))
print("N:{}".format(N))

### 连续特征
### 连续特征的联合概率如何处理？这里包含两种，
### 第一个是离散和连续的联合概率
### 第二个是连续和连续的联合概率
# MEAN = []
# STD = []
# for c in feature_maker.index2feature[-1].keys():
  # # print(c, continuous_X[Y==c,:])
  # MEAN.append(np.mean(continuous_X[Y==c,:]))
  # STD.append(np.std(continuous_X[Y==c,:]))
# print("mean:{} std:{}".format(MEAN, STD))



### test
test = [["青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.697, 0.460]]
# test = [["浅白", "硬挺", "浊响", "清晰", "凹陷", "硬滑", 0.697, 0.460]]
print("test:{}".format(json.dumps(test, ensure_ascii=False)))

test_X = feature_maker.encode(X=test, one_hot=True)
# test_X = X
print("encoded test:{}".format(test_X))

test_discrete = test_X[:,:-num_continuous_feature]
test_continuous = test_X[:,-num_continuous_feature:]
print("discrete test:{} continuous test:{}".format(test_discrete, test_continuous))

threshold_m = 0

pre = np.zeros((test_X.shape[0], num_class))
for k,x in enumerate(test_discrete):
  # print("k:{} x:{}".format(k, x))
  xx = np.reshape(x, (x.shape[0],1))
  N_mask = np.dot(xx, xx.T)
  # print(N_mask)
  for c in range(num_class):
    N_c = np.multiply(N[c], N_mask)
    # print("N_{}:{}".format(c, N_c))
    p = 1.0
    for i,fi in enumerate(x):
      if fi>0:
        #############################
        ### 先计算先验概率P(c,fi) ###
        #############################
        ### label为c并且特征f取值为i的个数
        d_fi = N_c[i][i]
        if d_fi<threshold_m:
          continue
        ### 特征f的属性个数
        fi_end = bisect.bisect_right(feature_partition, i)
        num_f_property = feature_partition[fi_end]-feature_partition[fi_end-1]
        prior_prob = (d_fi+1)*1.0/(num_sample+num_class*num_f_property)
        p *= prior_prob
        print("P(c:{}-{},i:{}-{}) = {}  d_fi:{}".format(c, feature_maker.index2feature[-1].get(c),
                                                        i, feature_maker.index2feature[fi_end-1].get(i-feature_partition[fi_end-1]), 
                                                        prior_prob, 
                                                        d_fi))
        #######################################
        ### 再计算条件概率 Σ log P(fj|c,fi) ###
        #######################################
        for j,fj in enumerate(N_c[i]):
          if fj>0 and j!=i:
            d_fj = fj
            fj_end = bisect.bisect_right(feature_partition, j)
            num_f_property = feature_partition[fj_end]-feature_partition[fj_end-1]
            cond_prob = (d_fj+1)*1.0/(d_fi+num_f_property)
            p *= cond_prob
            print("P(j:{}-{}|c:{}-{},i:{}-{}) = {}  d_fi:{} d_fj:{}".format(j, feature_maker.index2feature[fj_end-1].get(j-feature_partition[fj_end-1]), 
                                                               c, feature_maker.index2feature[-1].get(c),
                                                               i, feature_maker.index2feature[fi_end-1].get(i-feature_partition[fi_end-1]), 
                                                               cond_prob, 
                                                               d_fi, d_fj))
    print("k:{} c:{} p:{}".format(k, c, p))
    pre[k][c] += p
print("Predict:{}".format(pre))

# print(test_x)
# print(json.dumps(feature_maker.decode(test_x, one_hot=True), ensure_ascii=False))

