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
  ## wine dataset
  # data = np.loadtxt(ROOT_DIR+"/../wine/wine_data.csv", dtype=str, delimiter=",", skiprows=0)

  ## iris dataset
  # data = np.loadtxt(ROOT_DIR+"/../iris/iris_data.csv", dtype=str, delimiter=",", skiprows=0)

  ## car dataset
  data = np.loadtxt(ROOT_DIR+"/../car/car_data.csv", dtype=str, delimiter=",", skiprows=0)

  data = data[:,1:]

  print("... no-pruning ...")
  model = DecisionTreeModel()
  model.fit(data=data, algo_model="cart", split_ratio="9:1")
  # print(model.root["feature"])
  # print(json.dumps(model.root, indent=4, ensure_ascii=False))
  dot = model.export_graphviz()
  report = model.report_performances()
  print(report)

  ## pre_pruning
  print("... pre-pruning ...")
  pre_pruning_model = DecisionTreeModel()
  pre_pruning_model.fit(data=data, algo_model="cart", split_ratio="9:1", prune="pre")
  pre_pruning_dot = pre_pruning_model.export_graphviz(dot)
  pre_pruning_report = pre_pruning_model.report_performances()
  print(pre_pruning_report)

  ## post_pruning
  print("... post-pruning ...")
  post_pruning_model = DecisionTreeModel()
  post_pruning_model.fit(data=data, algo_model="cart", split_ratio="9:1", prune="post")
  post_pruning_dot = post_pruning_model.export_graphviz(pre_pruning_dot)
  # post_pruning_dot = post_pruning_model.export_graphviz()
  # dot.render(view=True, cleanup=True)
  # pre_pruning_dot.render(view=True, cleanup=True)
  post_pruning_dot.render(view=True, cleanup=True)
  post_pruning_report = post_pruning_model.report_performances()
  print(post_pruning_report)

