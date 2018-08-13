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
  ## x is (m, N_i-1), c should be (N_i-1, N_i), beta is (N_i,1) 
  ## return (m, N_i)
  @staticmethod
  def output(**kwargs):
    x = kwargs['x']
    ## set x.shape according (m, N_i, N_i-1) which is (m, 1, N_i-1)
    xx = np.reshape(x, (x.shape[0], 1, x.shape[1]))
    c = kwargs['c']
    ## set c.shape according (m, N_i, N_i-1) which is (1, N_i, N_i-1)
    cc = np.reshape(c, (1, c.shape[1], c.shape[0]))
    beta = kwargs['beta']
    # print("rdf output exp's index:{}".format(-1.0*beta*np.square(np.linalg.norm(xx-c, axis=2))))
    ## dist_square is (m, N_i)
    dist_square = np.square(np.linalg.norm(xx-cc, axis=2))
    # print("xx.shape:{} cc.shape:{} beta.shape:{} dist_square.shape{}".format(xx.shape, cc.shape, beta.shape, dist_square.shape))
    return np.exp(-1.0*np.multiply(dist_square, beta.T))

  ## derivative respect to x, return (m, N_i, N_i-1)
  ## x is (m, N_i-1), y is (m, N_i), so partial_y/partial_x is (m, N_i, N_i-1)
  @staticmethod
  def derivative_x(**kwargs):
    x = kwargs['x']
    c = kwargs['c']
    ## beta is (N_i, 1)
    beta = kwargs['beta']
    m = x.shape[0]
    ## set x.shape according (m, N_i, N_i-1) which is (m, 1, N_i-1)
    xx = np.reshape(x, (x.shape[0], 1, x.shape[1]))
    ## set c.shape according (m, N_i, N_i-1) which is (1, N_i, N_i-1)
    cc = np.reshape(c, (1, c.shape[1], c.shape[0]))
    ## set dist is (m, N_i)
    dist = np.linalg.norm(xx-cc, axis=2)
    ## set partial_prefix is (m, N_i, 1)
    partial_prefix = np.reshape(-2.0*np.multiply(RadialBasisFn.output(x=x, c=c, beta=beta), np.multiply(dist, beta.T)), 
                                (dist.shape[0], dist.shape[1], 1))
    ## set partial_prefix is (m, N_i-1, 1)
    partial_x = np.ones((x.shape[0], x.shape[1], 1))
    partial_y_x = np.array([np.dot(partial_prefix[i], partial_x[i].T) for i in range(m)])
    ## so return (m, N_i, N_i-1)
    return partial_y_x

  ## derivative respect to beta, return (m, N_i)
  ## y is (m, N_i), beta is (N_i, m), so partial_y/partial_beta is (m, N_i)
  @staticmethod
  def derivative_beta(**kwargs):
    x = kwargs['x']
    c = kwargs['c']
    ## beta is (N_i, 1)
    beta = kwargs['beta']
    m = x.shape[0]
    ## set x.shape according (m, N_i, N_i-1) which is (m, 1, N_i-1)
    xx = np.reshape(x, (x.shape[0], 1, x.shape[1]))
    ## set c.shape according (m, N_i, N_i-1) which is (1, N_i, N_i-1)
    cc = np.reshape(c, (1, c.shape[1], c.shape[0]))
    ## set dist_square is (m, N_i)
    dist_square = np.square(np.linalg.norm(xx-cc, axis=2))
    return -1.0*np.multiply(RadialBasisFn.output(x=x, c=c, beta=beta), dist_square)

  ## derivative respect to c, return (m, N_i, N_i-1)
  ## y is (m, N_i), c is (N_i-1, N_i), so partial_y/partial_c is (m, N_i)
  @staticmethod
  def derivative_c(**kwargs):
    x = kwargs['x']
    c = kwargs['c']
    ## beta is (N_i, 1)
    beta = kwargs['beta']
    m = x.shape[0]
    ## set x.shape according (m, N_i, N_i-1) which is (m, 1, N_i-1)
    xx = np.reshape(x, (x.shape[0], 1, x.shape[1]))
    ## set c.shape according (m, N_i, N_i-1) which is (1, N_i, N_i-1)
    cc = np.reshape(c, (1, c.shape[1], c.shape[0]))
    ## set dist is (m, N_i)
    dist = np.linalg.norm(xx-cc, axis=2)
    ## set partial_prefix is (m, N_i, 1)
    partial_prefix = np.reshape(2.0*np.multiply(RadialBasisFn.output(x=x, c=c, beta=beta), np.multiply(dist, beta.T)), 
                                (dist.shape[0], dist.shape[1], 1))
    ## set partial_prefix is (m, N_i-1, 1)
    partial_c = np.ones((x.shape[0], x.shape[1], 1))
    partial_y_c = np.array([np.dot(partial_prefix[i], partial_c[i].T) for i in range(m)])
    ## so return (m, N_i, N_i-1)
    return partial_y_c
    # return 2.0*np.multiply(RadialBasisFn.output(x=x, c=c, beta=beta), np.multiply(beta.T, dist))

if __name__ == '__main__':
  print(Sigmoid.output(0.))
  print(Sigmoid.derivative(0.))
