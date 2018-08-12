import sys
import numpy as np
from activate_fn import *
from loss import *
from learning_rate import *
from regularizor import *


class RBFNetwork:

  def __init__(self, topo, alpha, learning_rate=None, activate_fn=RadialBasisFn, loss_fn=LeastSqureLoss, lambdaa=0.0, regularization=Regularization):
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
    ### parameters
    ## beta.shape is (N_layer_i, sample_num(=1))
    self.beta = [np.random.normal(0, 1e-2, size=(self.topo[i],1)) for i in range(1, 1+len(self.topo[1:-1]))]
    ## c.shape is (N_layer_i, sample_num(=1), feature_num)
    self.C = [np.random.normal(0, 1e-2, size=(self.topo[i],1,self.topo[i-1])) for i in range(1, 1+len(self.topo[1:-1]))]
    ## only last layer is linerly connected
    self.W = np.random.normal(0, 1e-2, size=self.topo[-2:])
    ### parameters derivatives
    self.D_beta = [np.zeros(shape=(beta.shape)) for beta in self.beta]
    self.D_W = np.zeros(shape=(self.W.shape))
    ## forward temps: z.shape is (N_layer_i, sample_num)
    self.Z = [np.zeros(shape=(t,1)) for t in self.topo]
    ## backward temps
    ## eta = partial_E/partial_y : eta.shape is (N_layer_i, sample_num)
    self.eta = [np.zeros(shape=(t, 1)) for t in self.topo[1:]]
    print("W:{} beta:{} C:{} Z:{} eta:{}".format(len(self.W), len(self.beta), len(self.C), len(self.Z), len(self.eta)))
    return self

  ## X must be m x n where m is sample num and n is feature num
  def forward(self, X):
    ## input layer
    self.Z[0] = X.T
    # print("Z layer 0 : {}".format(self.Z[0].shape))
    for layer in range(1, len(self.Z)):
      ## last layer : liner output, no activate fn , N_layer x m
      if layer==len(self.Z)-1:
        self.Z[layer] = np.dot(self.W.T, self.Z[layer-1])
      else:
        self.Z[layer] = self.activate_fn.output(x=self.Z[layer-1].T, c=self.C[layer-1], beta=self.beta[layer-1])
      # print("Z layer {} : {}".format(layer, self.Z[layer].shape))
      # print("W layer {} : {}".format(layer-2, self.W[layer-2].shape))
      # if layer-1<len(self.beta):
        # print("beta layer {} : {}".format(layer-1, self.beta[layer-1].shape))
        # print("C layer {} : {}".format(layer-1, self.C[layer-1].shape))

  def backward(self, Y):
    ## the partial derivative to y of the output layer is: E=1/2(y-label)^2 => partial_E/partial_y = y-label
    partial_y = self.loss_fn.derivative(y=self.Z[-1], label=Y)
    ## eta = partial_E/partial_y
    self.eta[-1] = partial_y
    ## output layer W
    partial_reg_w = self.regularization.derivative(params=self.W)
    partial_w = (1.0-self.lambdaa)*np.dot(self.Z[-2], self.eta[-1].T) + self.lambdaa*partial_reg_w
    ## start from the last layer
    for layer in range(len(self.eta)-1, 0, -1):
      # print("backward Layer: {}".format(layer))
      partial_reg_beta = self.regularization.derivative(params=self.beta[layer-1])
      ## output layer to last hidden layer
      if layer==len(self.eta)-1:
        partial_beta = np.multiply(np.dot(self.W, self.eta[-1]), 
                                   self.activate_fn.derivative_beta(x=self.Z[layer-1].T, 
                                                                    c=self.C[layer-1], 
                                                                    beta=self.beta[layer-1]))
        self.eta[layer-1] = np.dot(self.W, self.eta[layer])
      ## i hidden layer to i-1 hidden layer
      else:
        partial_beta = np.multiply(self.eta[layer], 
                                   self.activate_fn.derivative_beta(x=self.Z[layer-1].T, 
                                                                    c=self.C[layer-1], 
                                                                    beta=self.beta[layer-1]))
        self.eta[layer-1] = self.activate_fn.derivative_x(x=self.Z[layer-1].T, 
                                                     c=self.C[layer-1], 
                                                     beta=self.beta[layer-1])
      # print("in backward, partial_beta.shape:{}".format(partial_beta.shape))
      ## sum along with sample num, return (N_layer_i, 1)
      partial_beta = np.reshape(np.sum(partial_beta, axis=1), (-1,1))
      # print("in backward, partial_beta.shape:{}".format(partial_beta.shape))
      partial_beta = (1.0-self.lambdaa)*partial_beta + self.lambdaa*partial_reg_beta
      ## parameters derivatives
      self.D_beta[layer-1] = partial_beta
    ## update parameters 
    delta = self.learning_rate.delta(derivatives=[self.D_W]+self.D_beta)
    delta_W = delta[0]
    delta_beta = delta[1:]
    self.W = self.W-delta_W
    self.beta = [self.beta[i]-delta_beta[i] for i in range(len(self.beta))]

  def loss(self, X, Y):
    self.forward(X)
    return (1.0-self.lambdaa)*self.loss_fn.output(y=self.Z[-1], label=Y) + self.lambdaa*self.regularization.output(params=[self.W]+self.beta)

  def predict(self, X):
    self.forward(X)
    # print("Z[-1].shape : {}".format(self.Z[-1].shape))
    return (self.Z[-1].T>0.5).astype(float)

  def accuracy(self, X, Y):
    p = self.predict(X)
    # print(np.equal(p,Y))
    return np.mean((np.mean(np.equal(p,Y), axis=1)==1).astype(float))

