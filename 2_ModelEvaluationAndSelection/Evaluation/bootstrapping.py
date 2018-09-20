#coding:utf-8
import numpy as np

def bootstrapping_indice(data, sample_size):
  m = data.shape[0]
  return np.random.choice(a=m, size=sample_size, replace=True)

