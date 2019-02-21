# coding:utf-8
import sys
import os

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR+"/../util")

import numpy as np
import matplotlib.pyplot as plt
# from PIL import Image
# import cPickle
# import pickle

import util

### 超参
H = 100 # size of hidden layer
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
M = N*K # number of examples
epoch = 10000
step_size = 1e-3
reg = 1e-3

### 生成数据
X, Y = util.make_spiral_data(N, D, K)
# lets visualize the data:
# plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)
# plt.show()
X = X.astype(np.float32)
print("X.shape:{} Y.shape:{}".format(X.shape, Y.shape))
print("X:{}".format(X[:5,:]))
print("Y:{}".format(Y[:5]))

### 初始化参数
W1 = 0.01 * np.random.randn(D, H)
b1 = np.zeros((1,H))
W2 = 0.01 * np.random.randn(H, K)
b2 = np.zeros((1,K))

for i in range(epoch):

  z1 = np.dot(X, W1) + b1 # M x H
  hidden = np.maximum(0.0, z1)  #relu
  z2 = np.dot(hidden, W2) + b2

  exp_z = np.exp(z2)
  probs = exp_z/np.sum(exp_z, axis=1, keepdims=True)
  nnl = -np.log(probs[range(M),Y]) # softmax cross entropy, nnl

  data_loss = np.sum(nnl)/M
  reg_loss = 0.5*reg*(np.sum(W2*W2)+np.sum(W1*W1))
  loss = data_loss + reg_loss

  if i%100 == 0:
    print("iteration {}: data loss:{}".format(i, data_loss))

  dz2 = probs
  dz2[range(M), Y] -= 1 # softmax的loss, Zj=c时是 Pj-1，Zj!=c时是Pj

  db2 = np.sum(dz2, axis=0, keepdims=True)
  dw2 = np.dot(hidden.T, dz2)
  dh = np.dot(dz2, W2.T) # backprob for the next layer

  dz1 = dh
  dz1[hidden<=0] = 0.0 # backprop relu
  dw1 = np.dot(X.T, dz1)
  db1 = np.sum(dz1, axis=0, keepdims=True)

  dw2 += reg*W2
  dw1 += reg*W1

  W2 -= step_size*dw2
  b2 -= step_size*db2
  W1 -= step_size*dw1
  b1 -= step_size*db1

def predict(X, W1, b1, W2, b2):
  h = np.maximum(0.0, np.dot(X, W1)+b1)
  scores = np.dot(h, W2) + b2
  predicted_class = np.argmax(scores, axis=1)
  return predicted_class

### 评测训练数据
predicted_class = predict(X, W1, b1, W2, b2)
print('training accuracy: %.2f' % (np.mean(predicted_class == Y)))

util.plot_decision_boundary(X, Y, lambda x:predict(x, W1, b1, W2, b2))
plt.show()
