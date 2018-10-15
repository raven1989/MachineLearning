# coding:utf-8

import sys
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

IDX_PACKAGE = 0
IDX_NAME = 1
IDX_CORP = 2
IDX_MAIN_CLASS = 3
IDX_SUB_CLASS = 4

fea_to_index = {}
index_to_fea = {}

app_dict = {}
# index_dict = {}
# index_desc = {}
with open("app_noted_new.utf8") as f:
  i = 0
  for line in f:
    sp = line.strip().split("\t")
    app = sp[IDX_PACKAGE]
    fea = "{}-{}".format(sp[IDX_PACKAGE], sp[IDX_NAME])
    # fea = sp[IDX_MAIN_CLASS]
    # fea = sp[IDX_SUB_CLASS]
    # rest = "\t".join(sp[1:])
    if fea not in fea_to_index:
      fea_to_index[fea] = i
      index_to_fea[i] = fea
      i += 1
    # else:
      # print("Duplicate feature:{}".format(fea))
    if app not in app_dict:
      app_dict[app] = fea
      # index_dict[i] = app
      # index_desc[i] = rest
print("Load app list, size:{}".format(len(app_dict)))
print("Load feature, size:{}".format(len(fea_to_index)))


m = 0
n = len(fea_to_index)
indptr = [0]
indices = []
data = []
labels = []
with open("guazi_app") as f:
# with open("test") as f:
  for line in f:
    sp = line.strip().split("\t")
    if len(sp) != 4:
      print(sp)
      continue
    imei, app_list, mobile, health = sp
    labels.append(1 if int(health)<6 else 0)
    fea_indices = []
    for app_data in app_list.split(";"):
      app = app_data.split(",")[0]
      if app not in app_dict:
        continue
      fea = app_dict[app]
      fea_indices.append(fea_to_index[fea])
    # print(fea_indices)
    # indices.append(fea_to_index[fea])
    # data.append(1)
    counter = Counter(fea_indices)
    indices.extend(counter.keys())
    data.extend(counter.values())
    indptr.append(len(indices))
    m += 1

print("m:{} n:{}".format(m, n))
# print("indptr:{}\nindices:{}".format(indptr, indices))


X = csr_matrix((data, indices, indptr), dtype=int, shape=(m,n))
Y = np.array(labels)
print("X:{}\nY:{}".format(X, Y))

# sys.exit(0)


### 卡方检验
def chi2_test(X, Y, kbest):
  # clf = SelectKBest(chi2, k='all')
  clf = SelectKBest(chi2, k=kbest)
  clf.fit(X, Y)
  selected = clf.get_support(indices=True)
  # print(selected)
  # print(clf.scores_[selected])
  sort_selected = sorted(range(len(selected)), key=clf.scores_[selected].take, reverse=True)
  sort_selected = selected[sort_selected]
  print(sort_selected)
  print(clf.scores_[sort_selected])
  return sort_selected

### RFE : Recursive feature elimination
def rfe(X, Y, kbest):
  m, n = X.shape
  svc = SVC(kernel="linear")
  rfe = RFE(estimator=svc, n_features_to_select=kbest, step=1, verbose=1)
  rfe.fit(X, Y)
  selected = filter(lambda i:rfe.support_[i], range(n))
  return selected

### L1正则
def l1_norm(X, Y):
  lsvc = LinearSVC(C=1., penalty="l1", dual=False, verbose=1, max_iter=5000).fit(X, Y)
  acc = lsvc.score(X, Y)
  print("SVC accuracy:{}".format(acc))
  model = SelectFromModel(lsvc, prefit=True)
  selected = model.get_support(indices=True)
  return selected

### 集成树
def ensemble_tree(X, Y):
  clf = ExtraTreesClassifier(verbose=1)
  clf.fit(X, Y)
  acc = clf.score(X, Y)
  print("Trees accuracy:{}".format(acc))
  model = SelectFromModel(clf, prefit=True)
  selected = model.get_support(indices=True)
  scores = clf.feature_importances_[selected]
  sort_selected = sorted(range(len(selected)), key=scores.take, reverse=True)
  sort_selected = selected[sort_selected]
  print("Feature imoprtances:{}".format(clf.feature_importances_[sort_selected]))
  return sort_selected

if __name__ == '__main__':
  # selected = chi2_test(X, Y, kbest=10)
  # selected = rfe(X, Y, kbest=10)
  # selected = l1_norm(X, Y)
  selected = ensemble_tree(X, Y)

  print("Selected feature num:{}".format(len(selected)))
  ### show selected features
  for idx in selected:
    print("{}".format(index_to_fea[idx]))

