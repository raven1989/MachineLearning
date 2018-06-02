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
  data = np.loadtxt(ROOT_DIR+"/../watermelon/watermelon_2.0.csv", dtype=str, delimiter=",", skiprows=0)
  data = data[:,1:]
  # print(data)
  model = DecisionTreeModel()
  model.fit(data=data, algo_model="cart")
  # print(model.root["feature"])
  print(json.dumps(model.root, indent=4, ensure_ascii=False))

  ## predict train
  x = data[1:,:-1]
  y = data[1:,-1]
  predictions = model.predict(x)
  print(",".join(predictions))
  print(",".join(y))

  dot = model.export_graphviz()
  print(dot.render(view=True, cleanup=True))
