# coding:utf-8
import sys
import numpy as np
import matplotlib.pyplot as plt

# N: number of points per class
# D: dimensionality
# K: number of classes
def make_spiral_data(N, D, K):
  X = np.zeros((N*K, D))
  Y = np.zeros(N*K, dtype=np.uint8)
  for j in xrange(K):
      ix = range(N*j,N*(j+1))
      r = np.linspace(0.0,1,N) # radius
      t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
      X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
      Y[ix] = j
  return X, Y

def plot_decision_boundary(X, Y, pred_func):
    # 设定最大最小值，附加一点点边缘填充
    x1_min, x1_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    x2_min, x2_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))

    # 用预测函数预测一下
    Z = pred_func(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)

    # 然后画出图
    plt.contourf(xx1, xx2, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral, edgecolors='k')

