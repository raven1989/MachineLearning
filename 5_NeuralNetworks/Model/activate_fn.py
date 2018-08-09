import sys
import numpy as np

class Sigmoid:
  @staticmethod
  def output(z):
    return 1.0/(1.0+np.exp(-1.0*z))
  @staticmethod
  def derivative(z):
    return Sigmoid.output(z)*(1.0-Sigmoid.output(z))

if __name__ == '__main__':
  print(Sigmoid.output(0.))
  print(Sigmoid.derivative(0.))
