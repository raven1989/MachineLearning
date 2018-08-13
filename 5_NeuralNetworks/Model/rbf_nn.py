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
    ## beta.shape is (N_i, 1)
    self.beta = [np.random.normal(0, 1e-1, size=(self.topo[i],1)) for i in range(1, 1+len(self.topo[1:-1]))]
    ## c.shape is (N_i-1, 1)
    self.C = [np.random.normal(0, 1e-1, size=(self.topo[i-1],self.topo[i])) for i in range(1, 1+len(self.topo[1:-1]))]
    ## only last layer is linerly connected
    self.W = np.random.normal(0, 1e-1, size=self.topo[-2:])
    ### parameters derivatives
    self.D_beta = [np.zeros(shape=(beta.shape)) for beta in self.beta]
    self.D_W = np.zeros(shape=(self.W.shape))
    ## forward temps: z.shape is (N_layer_i, sample_num)
    self.Z = [np.zeros(shape=(t,1)) for t in self.topo]
    ## backward temps
    ## eta = partial_E/partial_y : eta.shape is (N_layer_i, sample_num)
    self.eta = [np.zeros(shape=(t, 1)) for t in self.topo[1:]]
    print("W:{} beta:{} C:{} Z:{} eta:{}".format(self.W.shape, len(self.beta), len(self.C), len(self.Z), len(self.eta)))
    return self

  ## X must be m x n where m is sample num and n is feature num(also input layer neuron num N_1)
  def forward(self, X):
    ## input layer
    self.Z[0] = X.T
    # print("Z layer 0 : {}".format(self.Z[0].shape))
    for layer in range(1, len(self.Z)):
      ## last layer : liner output, no activate fn , N_layer x m
      if layer==len(self.Z)-1:
        self.Z[layer] = np.dot(self.W.T, self.Z[layer-1])
      else:
        self.Z[layer] = self.activate_fn.output(x=self.Z[layer-1].T, c=self.C[layer-1], beta=self.beta[layer-1]).T
      # print("Z layer {} : {}".format(layer, self.Z[layer].shape))
      # print("W layer {} : {}".format(layer-2, self.W[layer-2].shape))
      # if layer-1<len(self.beta):
        # print("beta layer {} : {}".format(layer-1, self.beta[layer-1].shape))
        # print("C layer {} : {}".format(layer-1, self.C[layer-1].shape))

  def backward(self, Y):
    ## the partial derivative to y of the output layer is (N_last, m) 
    partial_y = self.loss_fn.derivative(y=self.Z[-1], label=Y)
    ## eta = partial_E/partial_y_last
    self.eta[-1] = partial_y
    ## output layer partial_W, should be (N_i-1, N_i)
    partial_reg_w = self.regularization.derivative(params=self.W)
    partial_w = (1.0-self.lambdaa)*np.dot(self.Z[-2], self.eta[-1].T) + self.lambdaa*partial_reg_w
    ## self.eta[-1] is (N_last, m), W is (N_last-1, N_last), partial_E/partial_y_last-1 should be (N_last-1, m)
    self.eta[-2] = np.dot(self.W, self.eta[-1])
    ## start from the last hidden layer
    for layer in range(len(self.eta)-1, 0, -1):
      # print("backward Layer: {}".format(layer))
      partial_reg_beta = self.regularization.derivative(params=self.beta[layer-1])
      partial_beta = np.multiply(self.eta[layer-1], 
                                 self.activate_fn.derivative_beta(x=self.Z[layer-1].T, c=self.C[layer-1], beta=self.beta[layer-1]).T)
      # print("in backward, partial_beta.shape:{}".format(partial_beta.shape))
      ## sum along with sample num, return (N_layer_i, 1)
      partial_beta = np.reshape(np.sum(partial_beta, axis=1), (-1,1))
      # print("in backward, partial_beta:{}".format(partial_beta))
      partial_beta = (1.0-self.lambdaa)*partial_beta + self.lambdaa*partial_reg_beta
      ## parameters derivatives
      self.D_beta[layer-1] = partial_beta
      ## update eta
      if layer>2:
        ## cur_layer_partial_y is (m, N_cur_layer, N_cur_layer-1)
        cur_layer_partial_y = self.activate_fn.derivative_x(x=self.Z[layer-1].T, c=self.C[layer-1], beta=self.beta[layer-1])
        m, N_layer_cur, N_layer_before = cur_layer_partial_y.shape
        before_layer_partial_y = np.reshape([np.dot(cur_layer_partial_y[i].T, self.eta[layer-1]) for i in range(cur_layer_partial_y.shape[0])], 
                                            (m, N_layer_before))
        ## set eta (N, m)
        self.eta[layer-2] = before_layer_partial.T 
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

