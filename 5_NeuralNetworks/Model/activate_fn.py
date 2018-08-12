import sys
import numpy as np

class Sigmoid:
  @staticmethod
  def output(**kwargs):
    z = kwargs['z']
    return 1.0/(1.0+np.exp(-1.0*z))
  @staticmethod
  def derivative(**kwargs):
    z = kwargs['z']
    return Sigmoid.output(z=z)*(1.0-Sigmoid.output(z=z))

class RadialBasisFn:
  ## x must be m x n where m is sample num and n is feature num
  @staticmethod
  # N_layer x m
  def output(**kwargs):
    x = kwargs['x']
    ## set x.shape according (N_nueron, m, feature_num) = (1, m, feature_num)
    xx = np.reshape(x, (1,x.shape[0],x.shape[1]))
    c = kwargs['c']
    beta = kwargs['beta']
    # print("xx.shape:{} c.shape:{} beta.shape:{}".format(xx.shape, c.shape, beta.shape))
    # print("rdf output exp's index:{}".format(-1.0*beta*np.square(np.linalg.norm(xx-c, axis=2))))
    return np.exp(-1.0*beta*np.square(np.linalg.norm(xx-c, axis=2)))
  ## derivative respect to x
  @staticmethod
  def derivative_x(**kwargs):
    x = kwargs['x']
    xx = np.reshape(x, (1,x.shape[0],x.shape[1]))
    c = kwargs['c']
    beta = kwargs['beta']
    return -2.0*RadialBasisFn.output(x=x, c=c, beta=beta)*beta*np.linalg.norm(xx-c, axis=2)
  ## derivative respect to beta
  @staticmethod
  def derivative_beta(**kwargs):
    x = kwargs['x']
    xx = np.reshape(x, (1,x.shape[0],x.shape[1]))
    c = kwargs['c']
    # c = np.reshape(kwargs['c'], (1,-1))
    beta = kwargs['beta']
    return -1.0*RadialBasisFn.output(x=x, c=c, beta=beta)*np.square(np.linalg.norm(xx-c, axis=2))
  ## derivative respect to c
  @staticmethod
  def derivative_c(**kwargs):
    x = kwargs['x']
    xx = np.reshape(x, (1,x.shape[0],x.shape[1]))
    c = kwargs['c']
    # c = np.reshape(kwargs['c'], (1,-1))
    beta = kwargs['beta']
    return 2.0*RadialBasisFn.output(x=x, c=c, beta=beta)*beta*np.linalg.norm(xx-c, axis=2)

if __name__ == '__main__':
  print(Sigmoid.output(0.))
  print(Sigmoid.derivative(0.))
