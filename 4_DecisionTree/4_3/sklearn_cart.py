#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import os
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR+"/../Model")
from decision_tree_model import DecisionTreeModel
import numpy as np
import json
import graphviz
from sklearn import tree

if __name__ == "__main__":
  # data = np.loadtxt(ROOT_DIR+"/../watermelon/watermelon_2.0.csv", dtype=str, delimiter=",", skiprows=0)
  data = np.loadtxt(ROOT_DIR+"/../watermelon/watermelon_3.0.csv", dtype=str, delimiter=",", skiprows=0)
  # data = np.loadtxt(ROOT_DIR+"/../watermelon/watermelon_3.0_x.csv", dtype=str, delimiter=",", skiprows=0)
  data = data[:,1:]
  # print(data)
  model = DecisionTreeModel()
  model.fit(data=data, algo_model="id3")
  # print(model.root["feature"])
  # print(json.dumps(model.root, indent=4, ensure_ascii=False))

  ## predict train
  x = data[1:,:-1]
  y = data[1:,-1]
  # x = x[8:9,:]
  # y = y[8:9]

  print("---------------my own.DecisionTreeClassifier----------------")
  predictions = model.predict(x)
  print("Y:{}".format(",".join(y)))
  print("Predictions:{}".format(",".join(predictions)))
  print("train accuracy:{}".format(model.accuracy(y, predictions)))

  # sys.exit(0)

  dot = model.export_graphviz()
  print(dot.render(view=True, cleanup=True))

  ### 我们来看看西瓜数据集到底如何
  print("---------------sklearn.tree.DecisionTreeClassifier----------------")
  X = model.data[:,:-1]
  Y = model.data[:,-1]
  print("X:{}\nY:{}".format(X, Y))

  clf = tree.DecisionTreeClassifier()
  clf.fit(X, Y)

  predictions = clf.predict(X)
  print("Predictions:{}".format(predictions))
  print("Accuracy:{}".format(np.mean(predictions==Y)))

  dot_data = tree.export_graphviz(clf, out_file=None)
  graph = graphviz.Source(dot_data)
  graph.render(view=True, cleanup=True)


