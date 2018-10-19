# coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import os

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
# sys.path.append(ROOT_DIR+"/../Model")
sys.path.append(ROOT_DIR+"/../../0_FeatureMaker")

import numpy as np
from feature_maker import FeatureMaker
import json
from sklearn.svm import LinearSVC
from sklearn.metrics import hinge_loss


src = ROOT_DIR+'/../../Data/iris/iris_data.csv'
feature_types = [1,1,1,1,0]
feature_maker = FeatureMaker(src=src, delimiter=',', types=feature_types, norm=True)
X, Y = feature_maker.make(skip_rows=1, skip_cols=1, one_hot=True)
### 将数据集做成二分类
Y = np.array([[1] if y[0]>1 else [-1] for y in Y])
# Y = Y.flatten()
# print("X:{}\nY:{}".format(X, Y))

Xtrain, Xtest, Ytrain, Ytest = feature_maker.train_test_split(X, Y, test_size=0.3)
Xmarked, Xunmarked, Ymarked, Yunmarked = feature_maker.train_test_split(Xtrain, Ytrain, test_size=1.0/7)
Ytest = Ytest.flatten()
Ymarked = Ymarked.flatten()
Yunmarked = Yunmarked.flatten()
print("Xmarked.shape:{} Ymarked.shape:{}\nXunmarked.shape:{} Yunmarked.shape:{}\nXtest.shape:{} Ytest.shape:{}".format(Xmarked.shape, Ymarked.shape, Xunmarked.shape, Yunmarked.shape, Xtest.shape, Ytest.shape))


def assign_labels(pseudo_label, epsilon):
  for i in range(len(pseudo_label)-1):
    for j in range(i+1, len(pseudo_label[i+1:])):
      if pseudo_label[i]*pseudo_label[j]<0 and epsilon[i]>0 and epsilon[j]>0 and epsilon[i]+epsilon[j]>2:
        print("trans i:{} j:{}".format(i,j))
        pseudo_label[i] *= -1
        pseudo_label[j] *= -1
        return True
  return False

### hyper
Cmarked = 1.0
Cunmarked = 0.001

clf = LinearSVC(loss='hinge', verbose=1).fit(Xmarked, Ymarked)
print("W:{} b:{}".format(clf.coef_, clf.intercept_))
Ypseudo = clf.predict(Xunmarked).tolist()

while Cunmarked<Cmarked:
  Xtrain = np.concatenate((Xmarked, Xunmarked))
  Ytrain = np.concatenate((Ymarked, Ypseudo))
  sw = np.concatenate((np.zeros(Xmarked.shape[0])+Cmarked, np.zeros(Xunmarked.shape[0])+Cunmarked))
  clf.fit(Xtrain, Ytrain, sample_weight=sw)
  ### 计算松弛因子
  epsilon = [hinge_loss([Yunmarked[i]], clf.decision_function([Xunmarked[i]])) for i in range(Xunmarked.shape[0])]
  print("epsilon:{}\npseudo Y:{}".format(epsilon, Ypseudo))
  max_trans = len(Ypseudo)*3
  cnt = 0
  while assign_labels(pseudo_label=Ypseudo, epsilon=epsilon) and cnt<max_trans:
    print("pseudo Y:{}".format(Ypseudo))
    Ytrain = np.concatenate((Ymarked, Ypseudo))
    clf.fit(Xtrain, Ytrain, sample_weight=sw)
    epsilon = [hinge_loss([Yunmarked[i]], clf.decision_function([Xunmarked[i]])) for i in range(Xunmarked.shape[0])]
    cnt += 1
  Cunmarked = min(Cunmarked*2, Cmarked)


# clf.decision_function()
acc = clf.score(Xtest, Ytest)
print("Accuracy:{}".format(acc))


