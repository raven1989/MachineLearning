import sys
import numpy as np

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

