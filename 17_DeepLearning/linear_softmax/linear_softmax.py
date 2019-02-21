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
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
M = N*K # number of examples
epoch = 200
# batch = 128
step_size = 1e-2
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


# ### 标准化
# mu = np.mean(X, axis=0)
# sigma = np.std(X, axis=0)
# X = (X-mu)/sigma
# print("X:{}".format(X[:5,:]))
# print("Y:{}".format(Y[:5]))


### 初始化参数
W = 0.01 * np.random.randn(D, K)
b = np.zeros((1,K))

### 梯度下降
for i in range(epoch):
      # Xbatch = X[start:end,:]
      # num_example = end-start

      z = np.dot(X, W) + b # M x K
      exp_z = np.exp(z)
      probs = exp_z/np.sum(exp_z, axis=1, keepdims=True)
      nnl = -np.log(probs[range(M), Y]) # - yi * log Pi

      data_loss = np.sum(nnl)/M
      reg_loss = 0.5*reg*np.sum(W*W)
      loss = data_loss + reg_loss

      if i%10==0:
        print("iteration {}: loss:{}".format(i, loss))

      dz = probs
      dz[range(M), Y] -= 1 # softmax的loss, Zj=c时是 Pj-1，Zj!=c时是Pj
      dw = np.dot(X.T, dz)
      db = np.sum(dz, axis=0, keepdims=True)
      
      dw += reg*W

      W -= step_size * dw
      b -= step_size * db

print(W)
print(b)

def predict(X, W, b):
  scores = np.dot(X, W) + b
  predicted_class = np.argmax(scores, axis=1)
  return predicted_class


### 评测训练数据
predicted_class = predict(X, W, b)
print('training accuracy: %.2f' % (np.mean(predicted_class == Y)))

util.plot_decision_boundary(X, Y, lambda x:predict(x, W, b))
plt.show()
