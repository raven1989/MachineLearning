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
  data = np.loadtxt(ROOT_DIR+"/../watermelon/watermelon_2.0_x.csv", dtype=str, delimiter=",", skiprows=0)
  data = data[:,1:]
  # print(data)
  # model = DecisionTreeModel()
  # model.fit(data=data, algo_model="id3")
  # model.fit(data=data, algo_model="cart")
  # model.fit(data=data, algo_model="cart", split_ratio="watermelon")
  # model.fit(data=data, algo_model="cart", split_ratio="9:1")
  # print(model.root["feature"])
  # print(json.dumps(model.root, indent=4, ensure_ascii=False))

  # dot = model.export_graphviz()
  # dot.render(view=True, cleanup=True)

  ## post_pruning
  post_pruning_model = DecisionTreeModel()
  post_pruning_model.fit(data=data, algo_model="cart", split_ratio="watermelon", prune="post")

  post_pruning_dot = post_pruning_model.export_graphviz()
  post_pruning_dot.render(view=True, cleanup=True)
