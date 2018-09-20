#coding:utf-8
import sys
import numpy as np

class WeakClassifier:
  def fit(self, data, distribution):
    raise NotImpletementError("Abstract method not impletemented")
  ### 规范化输出1和-1
  def predict(self, X):
    raise NotImpletementError("Abstract method not impletemented")
  def mapping_label(Y):
    raise NotImpletementError("Abstract method not impletemented")

class AdaBoostClassifier:

  def __init__(self, weak_classifiers):
    self.num_weaker = 0
    self.weak_classifiers = weak_classifiers
    self.alpha = []

  def fit(self, data, skip_rows=0, skip_cols=0):
    m, n = data.shape
    m -= skip_rows
    n -= skip_cols
    X = data[skip_rows:, skip_cols:-1]
    Y = data[skip_rows:, -1]
    ### 准备一个小值，以应对弱分类器错误率为0的情况 
    epsilon = 1e-10
    ### 初始化样本分布权值
    distribution = np.array([1.0/m]*m)
    # print(distribution)
    for t in range(len(self.weak_classifiers)):
      weak_classifier = self.weak_classifiers[t]
      real = weak_classifier.mapping_label(Y)
      # print("data:{}".format(data))
      weak_classifier.fit(data, distribution)
      pre = weak_classifier.predict(X)
      error = 1.0-np.mean(pre==real)
      print("t:{} error:{} detail:{}".format(t, error, pre==real))
      if error>0.5:
        # print("Weak classifier {} error:{}, training exit.".format(t, error))
        break
      alpha = 0.5*np.log((1.0-error+epsilon)/(error+epsilon))
      self.alpha.append(alpha)
      # print("t:{} alpha:{}\ndistribution:{}\nd_sum:{}".format(t, self.alpha[t], distribution, np.sum(distribution)))
      d = [distribution[i]*np.exp(-self.alpha[t]*pre[i]*real[i]) for i in range(m)]
      # print("d:{}".format(d))
      z = np.sum(d)
      distribution = d/z
    while len(self.weak_classifiers)>len(self.alpha):
      self.weak_classifiers.pop()

  def predict(self, X):
    weak_pres = [self.weak_classifiers[i].predict(X)*self.alpha[i] for i in range(len(self.alpha))]
    # print(weak_pres)
    return np.sign(np.sum(weak_pres, axis=0))

    
