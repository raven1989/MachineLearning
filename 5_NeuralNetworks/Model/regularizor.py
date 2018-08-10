import sys
import numpy as np

class Regularization:
  @staticmethod
  def output(**kwargs):
    return 0
  @staticmethod
  def derivative(**kwargs):
    return 0

class L2Regularization(Regularization):
  @staticmethod
  def output(**kwargs):
    params = kwargs['params']
    l2 = np.sum([np.square(np.linalg.norm(np.reshape(p, -1))) for p in params])
    return l2
  @staticmethod
  def derivative(**kwargs):
    params = kwargs['params']
    return params

