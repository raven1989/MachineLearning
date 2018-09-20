#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import os
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR+"/../../4_DecisionTree/Model")
sys.path.append(ROOT_DIR+"/../Model")
from decision_tree_model import DecisionTreeModel
from bagging import WeakClassifier
from bagging import BaggingClassifier
import numpy as np
import json

class DecisionTreeAsWeakClassifier(WeakClassifier):
   
  def __init__(self, model, label_mapping):
    self.classifier = DecisionTreeModel()
    self.classifier.set_max_depth(2)
    self.model = model
    self.label_mapping = label_mapping

  def fit(self, data):
    # print(data)
    self.classifier.fit(data=data, algo_model=self.model)

  def predict(self, X):
    pre = self.classifier.predict(X)
    return np.array([self.label_mapping.get(y) for y in pre])

  def mapping_label(self, Y):
    return np.array([self.label_mapping.get(x) for x in Y])


if __name__ == "__main__":
  data = np.loadtxt(ROOT_DIR+"/../../4_DecisionTree/watermelon/watermelon_3.0_x.csv", dtype=str, delimiter=",", skiprows=0)
  # data = np.loadtxt(ROOT_DIR+"/../../4_DecisionTree/watermelon/watermelon_3.0.csv", dtype=str, delimiter=",", skiprows=0)
  data = data[:,1:]

  num_weak = 10
  num_sample = 27
  label_map = {"好瓜":1., "坏瓜":-1.}
  weak_classifiers = [DecisionTreeAsWeakClassifier(model="id3", label_mapping=label_map) for i in range(num_weak)]
  model = BaggingClassifier(weak_classifiers=weak_classifiers, num_sample=num_sample)

  model.fit(data, skip_rows=1)
  # print(json.dumps(model.root, indent=4, ensure_ascii=False))

  ## predict train
  x = data[1:,:-1]
  y = data[1:,-1]
  predictions = model.predict(x)
  real = weak_classifiers[0].mapping_label(y)
  print(predictions)
  print(real)
  print("Train Accuracy:{}".format(np.mean(predictions==real)))

  dot = None
  for weaker in model.weak_classifiers:
    dot = weaker.classifier.export_graphviz(dot)
  print(dot.render(view=True, cleanup=True))

