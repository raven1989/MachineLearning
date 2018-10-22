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
# src = ROOT_DIR+'/../../Data/iris/iris_data.csv'
# feature_types = [1,1,1,1,0]
feature_maker = FeatureMaker(src=src, delimiter=',', types=feature_types, norm=True)
X, Y = feature_maker.make(skip_rows=1, skip_cols=1, one_hot=True)
### 将数据集做成二分类
X = X[Y[:,0]!=1,:]
Y = Y[Y[:,0]!=1,:]
# print(Y)
Y = np.array([[1] if y[0]>1 else [-1] for y in Y])
# Y = Y.flatten()
print("X.shape:{}\nY.shape:{}".format(X.shape, Y.shape))

Xtrain, Xtest, Ytrain, Ytest = feature_maker.train_test_split(X, Y, test_size=0.3)
Xmarked, Xunmarked, Ymarked, Yunmarked = feature_maker.train_test_split(Xtrain, Ytrain, test_size=6.0/7)
Ytest = Ytest.flatten()
Ymarked = Ymarked.flatten()
Yunmarked = Yunmarked.flatten()
print("Xmarked.shape:{} Ymarked.shape:{}\nXunmarked.shape:{} Yunmarked.shape:{}\nXtest.shape:{} Ytest.shape:{}".format(Xmarked.shape, Ymarked, Xunmarked.shape, Yunmarked, Xtest.shape, Ytest))

# sys.exit(0)

def my_hinge(z):
  return max(0.0, np.mean(1.0-z))

### 每次找到最大的两个交换
def assign_labels(pseudo_label, epsilon):
  pos_max = 0.0
  neg_max = 0.0
  max_i = -1
  max_j = -1
  for i,e in enumerate(epsilon):
    if pseudo_label[i]>0 and (max_i<0 or e>pos_max):
      max_i = i
      pos_max = e
    elif pseudo_label[i]<0 and (max_j<0 or e>neg_max):
      max_j = i
      neg_max = e
    else:
      continue
  if max_i>0 and max_j>0 and pos_max>0 and neg_max>0 and pos_max+neg_max>2.0:
    print("epsilon:{}\npos max i:{} neg max j:{}".format(epsilon, max_i, max_j))
    pseudo_label[max_i] *= -1
    pseudo_label[max_j] *= -1
    return True
  return False

### hyper
Cmarked = 1.0
Cunmarked = 0.01

clf = LinearSVC(penalty='l2', loss='hinge', max_iter=2000, verbose=0).fit(Xmarked, Ymarked)
supervised_acc = clf.score(Xtest, Ytest)
print("W:{} b:{}".format(clf.coef_, clf.intercept_))

while Cunmarked<Cmarked:
  Ypseudo = clf.predict(Xunmarked).tolist()
  print("Seudo Y acc:{}".format(np.mean(Ypseudo==Yunmarked)))
  Xtrain = np.concatenate((Xmarked, Xunmarked))
  Ytrain = np.concatenate((Ymarked, Ypseudo))
  sw = np.concatenate((np.zeros(Xmarked.shape[0])+Cmarked, np.zeros(Xunmarked.shape[0])+Cunmarked))
  # print("Train label:{} Sample weight:{}".format(Ytrain, sw))
  # clf.fit(Xtrain, Ytrain, sample_weight=sw)
  clf = LinearSVC(penalty='l2', loss='hinge', max_iter=2000, verbose=0).fit(Xtrain, Ytrain, sample_weight=sw)
  # pre = [np.mean(clf.decision_function([Xunmarked[i]])) for i in range(Xunmarked.shape[0])]
  # my_pre = [np.mean(np.dot(clf.coef_, Xunmarked[i])+clf.intercept_) for i in range(Xunmarked.shape[0])]
  # print("sklearn pre:{}\nmy      pre:{}".format(pre, my_pre))
  ### 计算松弛因子
  # epsilon = np.array([hinge_loss([Ypseudo[i]], clf.decision_function([Xunmarked[i]])) for i in range(Xunmarked.shape[0])])
  # my_epsilon = [my_hinge(Ypseudo[i]*(np.dot(clf.coef_, Xunmarked[i])+clf.intercept_)) for i in range(Xunmarked.shape[0])]
  my_epsilon = np.array([my_hinge(Ypseudo[i]*clf.decision_function([Xunmarked[i]])) for i in range(Xunmarked.shape[0])])
  # print("epsilon   :{}\nmy epsilon:{}".format(["%.4f" % e for e in epsilon[epsilon[:]>2]], ["%.4f" % e for e in my_epsilon[my_epsilon[:]>2]]))
  # print("epsilon   :{}\nmy epsilon:{}".format(["%.3f" % round(e,3) for e in epsilon], ["%.3f" % round(e,3) for e in my_epsilon]))
  # max_trans = len(Ypseudo)*10
  # cnt = 0
  # sys.exit(0)
  while assign_labels(pseudo_label=Ypseudo, epsilon=my_epsilon): #and cnt<max_trans:
    old = Ytrain
    Ytrain = np.concatenate((Ymarked, Ypseudo))
    a, b = filter(lambda i:old[i]!=Ytrain[i], range(len(old)))
    print("reassign {}:{}, my epsilon {}:{}, old {}:{}, new {}:{}".format(a, b, my_epsilon[a-len(Ymarked)], my_epsilon[b-len(Ymarked)], old[a], old[b], Ytrain[a], Ytrain[b]))
    # clf.fit(Xtrain, Ytrain, sample_weight=sw)
    clf = LinearSVC(penalty='l2', loss='hinge', max_iter=2000, verbose=0).fit(Xtrain, Ytrain, sample_weight=sw)
    # epsilon = [hinge_loss([Ypseudo[i]], clf.decision_function([Xunmarked[i]])) for i in range(Xunmarked.shape[0])]
    # my_epsilon = [my_hinge(Ypseudo[i]*(np.dot(clf.coef_, Xunmarked[i])+clf.intercept_)) for i in range(Xunmarked.shape[0])]
    my_epsilon = [my_hinge(Ypseudo[i]*clf.decision_function([Xunmarked[i]])) for i in range(Xunmarked.shape[0])]
    # cnt += 1
    # if cnt > 5:
      # break
  Cunmarked = min(Cunmarked*2, Cmarked)
  # break


# clf.decision_function()
acc = clf.score(Xtest, Ytest)
print("Supervised accuracy:{}\nSemi accuracy:{}".format(supervised_acc, acc))


