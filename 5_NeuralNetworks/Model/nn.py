import sys
import numpy as np

class Sigmoid:
  @staticmethod
  def output(z):
    return 1.0/(1.0+np.exp(-1.0*z))
  @staticmethod
  def derivative(z):
    return Sigmoid.output(z)*(1.0-Sigmoid.output(z))

class LeastSqureLoss:
  @staticmethod
  def output(y, label):
    dist = np.reshape(y,-1) - np.reshape(label,-1)
    return 0.5 * np.dot(dist.T, dist)
  @staticmethod
  def derivative(y, label):
    return y-label

class NeuralNetwork:
  def __init__(self, topo, alpha, activate_fn=Sigmoid, loss_fn=LeastSqureLoss):
    self.topo = topo
    self.alpha = alpha
    self.activate_fn = activate_fn
    self.loss_fn = loss_fn
  def initialize(self):
    if len(self.topo)<2:
      raise ValueError("NeuralNetwork.topo must have 2 levels at least.")
    ## parameters
    self.W = [np.random.normal(0, 1, size=self.topo[i-1:i+1]) for i in range(1, len(self.topo))]
    self.b = [np.random.normal(0, 1, size=(self.topo[i],1)) for i in range(1, len(self.topo))]
    ## forward temps
    self.Z = [np.zeros(shape=(t,1)) for t in self.topo]
    # self.Y = [np.zeros(shape=(t,1)) for t in self.topo]
    ## backward temps
    ## eta = partial_E/partial_y * partial_y/partial_z = partial_E/partial_y * g(z) * (1-g(z))
    self.eta = [np.zeros(shape=(t, 1)) for t in self.topo[1:]]
    print("W:{} b:{} Z:{} eta:{}".format(len(self.W), len(self.b), len(self.Z), len(self.eta)))
    return self
  def forward(self, X):
    # self.Z[0] = np.dot(self.W[0].T, np.reshape(X, (self.topo[0],1))) + self.b[0]
    self.Z[0] = np.reshape(X, self.Z[0].shape)
    for layer in range(1, len(self.Z)):
      self.Z[layer] = np.dot(self.W[layer-1].T, self.activate_fn.output(self.Z[layer-1])) + self.b[layer-1]
    # print("y: {}".format(self.activate_fn.output(self.Z[-1])))
  def backward(self, Y):
    ## the partial derivative to y of the output layer is: E=1/2(y-label)^2 => partial_E/partial_y = y-label
    # partial_y = self.activate_fn.output(self.Z[-1]) - np.reshape(Y, self.Z[-1].shape)
    partial_y = self.loss_fn.derivative(self.activate_fn.output(self.Z[-1]), np.reshape(Y, self.Z[-1].shape))
    # loss = self.loss_fn.output(self.activate_fn.output(self.Z[-1]), Y)
    # print("loss: {}".format(loss))
    ## eta = partial_E/partial_y * partial_y/partial_z = partial_E/partial_y * g(z) * (1-g(z))
    self.eta[-1] = np.multiply(partial_y, self.activate_fn.derivative(self.Z[-1]))
    ## start from the last layer
    for layer in range(len(self.eta)-1, -1, -1):
      # print("Layer: {}".format(layer))
      partial_w = np.dot(self.activate_fn.output(self.Z[layer]), self.eta[layer].T)
      partial_b = self.eta[layer]
      ## update eta for next layer of backward propagation 
      if layer>0:
        partial_y = np.dot(self.W[layer], self.eta[layer])
        self.eta[layer-1] = np.multiply(partial_y, self.activate_fn.derivative(self.Z[layer]))
      ## update parameters 
      self.W[layer] -= self.alpha * partial_w
      self.b[layer] -= self.alpha * partial_b
  def loss(self, X, Y):
    self.forward(X)
    return self.loss_fn.output(self.activate_fn.output(self.Z[-1]), Y)
  def predict(self, X):
    self.forward(X)
    return self.activate_fn.output(self.Z[-1])

if __name__ == '__main__':
  print(Sigmoid.output(0.))
  print(Sigmoid.derivative(0.))
