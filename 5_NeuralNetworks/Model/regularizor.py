import sys
import numpy as np

class Regularization:
  @staticmethod
  def output(*args):
    return 0
  @staticmethod
  def derivative(p):
    return 0

class L2Regularization(Regularization):
  @staticmethod
  def output(*args):
    l2 = np.sum([np.square(np.linalg.norm(np.reshape(p, -1))) for p in args])
    return l2
  @staticmethod
  def derivative(p):
    return p

