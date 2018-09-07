#! coding:utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import os
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR+"/../../0_FeatureMaker")
import numpy as np

def sigmoid(z):
  return 1.0/(1.0+np.e**-z)

### beta = [W; b] = [w1; w2; ...; wi; ...; wn; b] shape:(n+1,1)
### x = [X; 1] = [x1; x2; ...; xi; ...; xn; 1] shape(n+1,1)
### so, beta.T dot x = W.T dot x + b

### probability for y=1 : p(y=1;x)
def p_1(beta, x):
  return sigmoid(np.dot(beta.T, x))

def log_likelihood_first_second_derivatives(beta, x, y):
  p1 = p_1(beta, x)
  first_derivative = np.multiply(x,(p1-y))
  second_derivative = np.dot(x, x.T*p1*(1-p1))
  # print("p1:{} first derivative.shape:{} second derivative.shape:{}".format(p1, first_derivative.shape, second_derivative.shape))
  return first_derivative, second_derivative

def gussian_kernel(sigma, x1, x2):
  return np.exp(-1.0/sigma*(np.linalg.norm(x1-x2))**2)

def gussian_kernel_matrix(X, sigma):
  m, n = X.shape
  K = []
  for i in range(m):
    for j in range(m):
      K.append(gussian_kernel(sigma, X[i], X[j]))
  K = np.reshape(K, (m,m))
  return K

def log_likelihood(beta, K, Y):
  m = Y.shape[0]
  l = 0
  for i in range(m):
    # print(np.dot(beta.T, K[:,i:i+1]))
    line = np.dot(beta.T, K[:,i:i+1])
    l += -1.0*Y[i]*line + np.log(1.0+np.e**line)
    # print("line:{} l:{}".format(line, l))
  return l

def reference(beta, sigma, X, x):
  m, _ = X.shape
  num, _ = x.shape
  print("X.shape:{} x.shape:{}".format(X.shape, x.shape))
  pres = []
  for j in range(num):
    p = []
    for i in range(m):
      p.append(gussian_kernel(sigma, x[j:j+1,:], X[i:i+1,:]))
    p.append(1)
    p = np.reshape(p, (m+1,-1))
    p = sigmoid(np.dot(beta.T, p))
    pres.append(p)
  return np.reshape(pres, (x.shape[0],1))

def predict(beta, sigma, X, x):
  pre = reference(beta, sigma, X, x)>0.5
  return np.array(pre, dtype=float)

def cal_accuracy(predictions, true_ys):
  pre = np.reshape(predictions, true_ys.shape)
  correct = np.equal(true_ys, pre)
  accuracy = np.sum([1 if x else 0 for x in correct])*1.0/len(correct)
  return accuracy

if __name__=='__main__':
  csv = ROOT_DIR+'/../../Data/watermelon/watermelon_3.0_x.csv'
  data = np.loadtxt(csv, dtype=float, delimiter=",")
  X = data[1:, 1:-1]
  Y = data[1:, -1]

  ### hyper parameters
  sigma = 0.5
  epsilon = 1e-6

  ### parameters
  # alpha = np.random.normal(0, 0.1, (X.shape[0],1))
  # b = np.random.normal(0, 0.1)

  alpha = np.zeros((X.shape[0],1))
  b = 1

  beta = np.reshape(np.append(alpha, b), (alpha.shape[0]+1, -1))
  print("Parameter alpha.shape:{} beta.shape:{}".format(alpha.shape, beta.shape))

  K = gussian_kernel_matrix(X, sigma)
  K = np.row_stack((K, np.ones((1,K.shape[1]))))
  K = np.column_stack((K, np.ones((K.shape[0],1))))
  # print("Kernel K:{}".format(K))

  m, _ = X.shape
  epoch = 0
  l = 0
  while True:
    first = 0;
    second = 0;
    for i in range(m):
      first_derivative, second_derivative = log_likelihood_first_second_derivatives(beta, K[:,i:i+1], Y[i])
      first = first + first_derivative
      second = second + second_derivative
    # print("beta:{} first:{} second:{}".format(beta.shape, first.shape, second.shape))
    ### 高维下，二阶导数是海森矩阵，相应的更新公式是 beta(t+1) = beta(t) - H.inv dot first_derivative
    beta = beta - np.dot(np.linalg.inv(second), first)
    cur_l = log_likelihood(beta, K, Y)
    if epoch%1==0:
      print("Epoch:{} log likelihood:{}".format(epoch, cur_l))
    if np.absolute(l-cur_l)<epsilon:
      break
    l = cur_l
    epoch += 1
  l = log_likelihood(beta, K, Y)
  print("End log likelihood:{}".format(l))
  ### predict
  predictions = predict(beta, sigma, X, X)
  print("Predictions:{} Y:{}".format(predictions, Y))
  acc = cal_accuracy(predictions, Y)
  print("Accuracy:{}".format(acc))

