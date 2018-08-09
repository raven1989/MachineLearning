import sys
import numpy as np

#acumulated square loss
class LeastSqureLoss:
  ## y.shape should be m x N where m is num of samples
  @staticmethod
  def output(y, label):
    shape = y.shape
    if len(shape)<2:
      shape = (1,shape[0])
    m, n = shape
    dist = np.reshape(y,shape) - np.reshape(label,shape)
    return np.mean(np.sum(0.5*np.square(dist), axis=1))
  @staticmethod
  def derivative(y, label):
    shape = y.shape
    if len(shape)<2:
      shape = (1,shape[0])
    m, n = shape
    return 1.0/m * (np.reshape(y,shape)-np.reshape(label,shape))

