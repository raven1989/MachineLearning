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

if __name__ == "__main__":
  # data = np.loadtxt(ROOT_DIR+"/../watermelon/watermelon_2.0.csv", dtype=str, delimiter=",", skiprows=0)
  # data = np.loadtxt(ROOT_DIR+"/../watermelon/watermelon_3.0.csv", dtype=str, delimiter=",", skiprows=0)
  data = np.loadtxt(ROOT_DIR+"/../watermelon/watermelon_3.0_x.csv", dtype=str, delimiter=",", skiprows=0)
  data = data[:,1:]
  ###测试data weights support，样本权重设置为1/m，则结果应该和不设置一样
  m, n = data.shape
  m -= 1
  data_weights = np.array([1.0/m]*m)
  ###测试data weights support，没有样本权重，则结果应该和设置为1/m一样
  data_weights = None
  # print(data)
  model = DecisionTreeModel()
  model.fit(data=data, algo_model="id3", data_weights=data_weights)
  # print(model.root["feature"])
  print(json.dumps(model.root, indent=4, ensure_ascii=False))

  ## predict train
  x = data[1:,:-1]
  y = data[1:,-1]
  predictions = model.predict(x)
  print(",".join(predictions))
  print(",".join(y))
  print("train accuracy:{}".format(model.accuracy(y, predictions)))

  dot = model.export_graphviz()
  print(dot.render(view=True, cleanup=True))
