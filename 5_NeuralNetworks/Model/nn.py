import sys
import numpy as np

class Sigmoid:
  @staticmethod
  def output(z):
    return 1.0/(1.0+np.exp(-1.0*z))
  @staticmethod
  def derivative(z):
    return Sigmoid.output(z)*(1.0-Sigmoid.output(z))

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

class LearningRate(object):
  def __init__(self, learning_rate):
    self.learning_rate = learning_rate
  def delta(self, derivatives):
    return [self.learning_rate*d for d in derivatives]

class MomentumLearningRate(LearningRate):
  def __init__(self, learning_rate, beta):
    super(MomentumLearningRate, self).__init__(learning_rate)
    self.beta = beta
    self.v = None
  def delta(self, derivatives):
    if self.v is None:
      self.v = [np.zeros(d.shape) for d in derivatives]
    self.v = [self.beta*self.v[i] + self.learning_rate*derivatives[i] for i in range(len(derivatives))]
    return self.v

class NeuralNetwork:
  def __init__(self, topo, alpha, learning_rate=None, activate_fn=Sigmoid, loss_fn=LeastSqureLoss, lambdaa=0.0, regularization=Regularization):
    self.topo = topo
    self.alpha = alpha
    self.learning_rate = LearningRate(alpha) if learning_rate is None else learning_rate
    self.lambdaa = lambdaa
    self.activate_fn = activate_fn
    self.loss_fn = loss_fn
    self.regularization = regularization
  def initialize(self):
    if len(self.topo)<2:
      raise ValueError("NeuralNetwork.topo must have 2 levels at least.")
    ## parameters
    self.W = [np.random.normal(0, 1, size=self.topo[i-1:i+1]) for i in range(1, len(self.topo))]
    self.b = [np.random.normal(0, 1, size=(1,self.topo[i])) for i in range(1, len(self.topo))]
    ## parameters derivatives
    self.D_W = [np.zeros(shape=(w.shape)) for w in self.W]
    self.D_b = [np.zeros(shape=(b.shape)) for b in self.b]
    ## forward temps
    self.Z = [np.zeros(shape=(t,1)) for t in self.topo]
    # self.Y = [np.zeros(shape=(t,1)) for t in self.topo]
    ## backward temps
    ## eta = partial_E/partial_y * partial_y/partial_z = partial_E/partial_y * g(z) * (1-g(z))
    self.eta = [np.zeros(shape=(t, 1)) for t in self.topo[1:]]
    print("W:{} b:{} Z:{} eta:{}".format(len(self.W), len(self.b), len(self.Z), len(self.eta)))
    return self
  ## X must be m x n where m is sample num and n is feature num
  def forward(self, X):
    # self.Z[0] = np.dot(self.W[0].T, np.reshape(X, (self.topo[0],1))) + self.b[0]
    self.Z[0] = X
    # print("Z layer 0 : {}".format(self.Z[0].shape))
    for layer in range(1, len(self.Z)):
      if layer==1:
        self.Z[layer] = np.dot(self.Z[layer-1], self.W[layer-1]) + self.b[layer-1]
      else:
        self.Z[layer] = np.dot(self.activate_fn.output(self.Z[layer-1]), self.W[layer-1]) + self.b[layer-1]
      # print("W layer {} : {}".format(layer-1, self.W[layer-1].shape))
      # print("b layer {} : {}".format(layer-1, self.b[layer-1].shape))
      # print("Z layer {} : {}".format(layer, self.Z[layer].shape))
  def backward(self, Y):
    ## the partial derivative to y of the output layer is: E=1/2(y-label)^2 => partial_E/partial_y = y-label
    # partial_y = self.activate_fn.output(self.Z[-1]) - np.reshape(Y, self.Z[-1].shape)
    partial_y = (1-self.lambdaa) * self.loss_fn.derivative(self.activate_fn.output(self.Z[-1]), Y)
    ## eta = partial_E/partial_y * partial_y/partial_z = partial_E/partial_y * g(z) * (1-g(z))
    self.eta[-1] = np.multiply(partial_y, self.activate_fn.derivative(self.Z[-1]))
    ## start from the last layer
    for layer in range(len(self.eta)-1, -1, -1):
      # print("Layer: {}".format(layer))
      partial_reg_w = self.regularization.derivative(self.W[layer])
      partial_reg_b = self.regularization.derivative(self.b[layer])
      partial_w = (1-self.lambdaa)*np.dot(self.activate_fn.output(self.Z[layer]).T, self.eta[layer]) + self.lambdaa*partial_reg_w
      partial_b = (1-self.lambdaa)*np.sum(self.eta[layer], axis=0) + self.lambdaa*partial_reg_b
      ## update eta for next layer of backward propagation 
      if layer>0:
        partial_y = np.dot(self.eta[layer], self.W[layer].T)
        self.eta[layer-1] = np.multiply(partial_y, self.activate_fn.derivative(self.Z[layer]))
      ## parameters derivatives
      self.D_W[layer] = partial_w
      self.D_b[layer] = partial_b
    ## update parameters 
    delta = self.learning_rate.delta(self.D_W+self.D_b)
    delta_W = delta[:len(self.D_W)]
    delta_b = delta[len(self.D_W):]
    self.W = [self.W[i]-delta_W[i] for i in range(len(self.W))]
    self.b = [self.b[i]-delta_b[i] for i in range(len(self.b))]
  def loss(self, X, Y):
    self.forward(X)
    return (1.0-self.lambdaa)*self.loss_fn.output(self.activate_fn.output(self.Z[-1]), Y) + self.lambdaa*self.regularization.output(*(self.W+self.b))
  def predict(self, X):
    self.forward(X)
    return (self.activate_fn.output(self.Z[-1])>0.5).astype(float)
  def accuracy(self, X, Y):
    p = self.predict(X)
    return np.mean((np.mean(np.equal(p,Y), axis=1)==1).astype(float))

if __name__ == '__main__':
  print(Sigmoid.output(0.))
  print(Sigmoid.derivative(0.))
