#coding:utf-8
import sys
import os
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR+"/../Model")
from decision_tree_model import DecisionTreeModel
import numpy as np
import json

if __name__ == "__main__":
  data = np.loadtxt(ROOT_DIR+"/../watermelon/watermelon_id3.csv", dtype=str, delimiter=",", skiprows=0)
  data = data[:,1:]
  model = DecisionTreeModel()
  model.fit(data=data, algo_model="id3")
  # print(model.root["feature"])
  print(json.dumps(model.root, indent=2))
