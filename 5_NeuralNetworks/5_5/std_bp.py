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
feature_maker = FeatureMaker(src=src, delimiter=',', types=feature_types, norm=True)
X, Y = feature_maker.make(skip_rows=1, skip_cols=1)
print("X : {}".format(X))
print("Y : {}".format(Y))

topo = [np.reshape(X[0],-1).shape[0], 5, 1]
alpha = 1
print("nn topo : {}".format(topo))
network = NeuralNetwork(topo=topo, alpha=alpha).initialize()

for epoch in range(100000):
  for x,y in zip(X,Y):
    network.forward(np.reshape(x, newshape=(1,x.shape[0])))
    network.backward(np.reshape(y, newshape=(1,y.shape[0])))
  if epoch % 1000 == 0:
    # loss = []
    # for x,y in zip(X,Y):
    #   loss.append(network.loss(x, y))
    loss = network.loss(X, Y)
    print("Epoch:{} Training Loss:{}".format(epoch, np.mean(loss)))

# pre = []
# for x in X:
#   pre.append(network.predict(x))
# pre = (np.reshape(pre, Y.shape)>0.5).astype(int)
pre = (network.predict(X)>0.5).astype(float)
print("Predict : {}".format(pre))

