#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import os
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR+"/../Model")
sys.path.append(ROOT_DIR+"/../../0_FeatureMaker")
from feature_maker import FeatureMaker
from nn import NeuralNetwork
import numpy as np


src = ROOT_DIR+'/../../Data/watermelon/watermelon_3.0.csv'
feature_types = [0,0,0,0,0,0,1,1,0]
feature_maker = FeatureMaker(src=src, delimiter=',', types=feature_types, norm=False)
X, Y = feature_maker.make(skip_rows=1, skip_cols=1)
print("X : {}".format(X))
print("Y : {}".format(Y))

topo = [np.reshape(X[0],-1).shape[0], 10, 1]
alpha = 0.01
print("nn topo : {}".format(topo))
network = NeuralNetwork(topo=topo, alpha=alpha).initialize()

for epoch in range(100):
  for x,y in zip(X,Y):
    network.forward(X[0])
    network.backward(Y[0])
  print("Loss : {}".format(network.loss(X[0], Y[0])))
