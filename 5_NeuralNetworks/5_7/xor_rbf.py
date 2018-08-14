#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import os
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR+"/../Model")
sys.path.append(ROOT_DIR+"/../../0_FeatureMaker")
import numpy as np
from feature_maker import FeatureMaker
from rbf_nn import RBFNetwork
from regularizor import L2Regularization
from learning_rate import MomentumLearningRate


src = ROOT_DIR+'/../../Data/xor/xor.csv'
feature_types = [0,0,0]
feature_maker = FeatureMaker(src=src, delimiter=',', types=feature_types, norm=True)
X, Y = feature_maker.make(skip_rows=1, skip_cols=1)
# print("X : {}".format(X))
# print("Y : {}".format(Y))

train_X, test_X, train_Y, test_Y = feature_maker.train_test_split(X, Y, test_size=0, random_state=1234)
# print("Test X : {}".format(test_X))
# print("Test Y : {}".format(test_Y))

train_X = np.reshape([[0,0],[0,1],[1,0],[1,1]], (-1,2))
train_Y = np.reshape([0,1,1,0], (-1,1))
print("Train X : {}".format(train_X))
print("Train Y : {}".format(train_Y))

N_input = np.reshape(train_X[0],-1).shape[0]
N_output = np.reshape(train_Y[0],-1).shape[0]
topo = [N_input, 10, N_output]
alpha = 0.01
lambdaa = 0.0001
print("nn topo : {}".format(topo))

## Momentum
learning_rate = MomentumLearningRate(learning_rate=alpha, beta=0.99)

network = RBFNetwork(topo=topo, init_std=1e-1, learning_rate=learning_rate, alpha=alpha, lambdaa=lambdaa, regularization=L2Regularization).initialize()

for epoch in range(2000):
  network.forward(train_X)
  network.backward(train_Y)
  if epoch % 100 == 0:
    train_loss = np.mean(network.loss(train_X, train_Y))
    # print("Epoch:{} Training Loss:{}".format(epoch, train_loss))
    # # # test_loss = np.mean(network.loss(test_X, test_Y))
    # # # test_acc = network.accuracy(test_X, test_Y)
    train_acc = network.accuracy(train_X, train_Y)
    print("Epoch:{} Training Loss:{} Train Acc:{}".format(epoch, train_loss, train_acc))

pre = network.predict(train_X)
print("Predict : {}".format(pre))

