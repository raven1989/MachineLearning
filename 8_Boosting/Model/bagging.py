# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np
from collections import Counter
import os
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR+"/../../2_ModelEvaluationAndSelection/Evaluation")
from bootstrapping import bootstrapping_indice

class WeakClassifier:
  def fit(self, data):
    raise NotImpletementError("Abstract method not impletemented")
  ### 规范化输出1和-1
  def predict(self, X):
    raise NotImpletementError("Abstract method not impletemented")
  def mapping_label(Y):
    raise NotImpletementError("Abstract method not impletemented")

class BaggingClassifier:

  def __init__(self, weak_classifiers, num_sample):
    self.weak_classifiers = weak_classifiers
    self.num_sample = num_sample

  def fit(self, data, skip_rows=0, skip_cols=0):
    m, n = data.shape
    m -= skip_rows
    n -= skip_cols
    X = data[skip_rows:, skip_cols:-1]
    Y = data[skip_rows:, -1]
    zero = np.zeros(1, dtype=int)
    for t,weak_classifier in enumerate(self.weak_classifiers):
      choice = bootstrapping_indice(X, self.num_sample)
      choice += skip_rows
      choice = np.concatenate((zero,choice))
      d = data[choice, :]
      print("t:{} choice:{}".format(t, choice))
      weak_classifier.fit(d)

  def predict(self, X):
    weak_pres = np.array([it.predict(X) for it in self.weak_classifiers]).T
    # print(weak_pres)
    pres = np.array([Counter(it).most_common(1)[0][0] for it in weak_pres])
    return pres

