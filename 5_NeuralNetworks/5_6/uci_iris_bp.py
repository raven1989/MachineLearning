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
from nn import NeuralNetwork
from regularizor import L2Regularization
from learning_rate import MomentumLearningRate


src = ROOT_DIR+'/../../Data/iris/iris_data.csv'
feature_types = [1,1,1,1,0]
feature_maker = FeatureMaker(src=src, delimiter=',', types=feature_types, norm=True)
X, Y = feature_maker.make(skip_rows=1, skip_cols=1)
# print("X : {}".format(X))
# print("Y : {}".format(Y))

train_X, test_X, train_Y, test_Y = feature_maker.train_test_split(X, Y, test_size=0.2, random_state=1234)
# train_X, test_X, train_Y, test_Y = train_X[:5,:], test_X[:5,:], train_Y[:5,:], test_Y[:5,:]
print("Train X : {}\n...".format(train_X[:5,:]))
print("Train Y : {}\n...".format(train_Y[:5,:]))
print("Test X : {}".format(test_X))
print("Test Y : {}".format(test_Y))

N_input = np.reshape(X[0],-1).shape[0]
N_output = np.reshape(Y[0],-1).shape[0]
topo = [N_input, 6, N_output]
alpha = 1
lambdaa = 0.0001
print("nn topo : {}".format(topo))

## Momentum
learning_rate = MomentumLearningRate(learning_rate=alpha, beta=0.9)

network = NeuralNetwork(topo=topo, alpha=alpha, learning_rate=learning_rate, lambdaa=lambdaa, regularization=L2Regularization).initialize()

for epoch in range(300):
  network.forward(train_X)
  network.backward(train_Y)
  if epoch % 10 == 0:
    train_loss = np.mean(network.loss(train_X, train_Y))
    test_loss = np.mean(network.loss(test_X, test_Y))
    test_acc = network.accuracy(test_X, test_Y)
    print("Epoch:{} Training Loss:{} Test Loss:{} Test Acc:{}".format(epoch, train_loss, test_loss, test_acc))
## final
train_loss = np.mean(network.loss(train_X, train_Y))
test_loss = np.mean(network.loss(test_X, test_Y))
test_acc = network.accuracy(test_X, test_Y)
print("Epoch:{} Training Loss:{} Test Loss:{} Test Acc:{}".format("Final", train_loss, test_loss, test_acc))

# pre = (network.predict(test_X)>0.5).astype(float)
pre = network.predict(test_X)
print("Predict : {}".format(pre))

