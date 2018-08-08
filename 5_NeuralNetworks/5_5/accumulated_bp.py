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
from nn import L2Regularization
import numpy as np


src = ROOT_DIR+'/../../Data/watermelon/watermelon_3.0.csv'
feature_types = [0,0,0,0,0,0,1,1,0]
feature_maker = FeatureMaker(src=src, delimiter=',', types=feature_types, norm=True)
X, Y = feature_maker.make(skip_rows=1, skip_cols=1)
# print("X : {}".format(X))
# print("Y : {}".format(Y))

train_X, test_X, train_Y, test_Y = feature_maker.train_test_split(X, Y, test_size=5, random_state=1234)
print("Train X : {}".format(train_X))
print("Train Y : {}".format(train_Y))
print("Test X : {}".format(test_X))
print("Test Y : {}".format(test_Y))

topo = [np.reshape(X[0],-1).shape[0], 5, 1]
alpha = 1
lambdaa = 0.001
print("nn topo : {}".format(topo))
network = NeuralNetwork(topo=topo, alpha=alpha, lambdaa=lambdaa, regularization=L2Regularization).initialize()

for epoch in range(9200):
  network.forward(train_X)
  network.backward(train_Y)
  if epoch % 100 == 0:
    train_loss = np.mean(network.loss(train_X, train_Y))
    test_loss = np.mean(network.loss(test_X, test_Y))
    print("Epoch:{} Training Loss:{} Test Loss:{}".format(epoch, train_loss, test_loss))

pre = (network.predict(test_X)>0.5).astype(float)
print("Predict : {}".format(pre))

