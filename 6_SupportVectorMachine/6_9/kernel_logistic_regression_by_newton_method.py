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
  first_derivative = x * (p1-y)
  second_derivative = np.dot(x.T, x)*p1*(1-p1)
  return first_derivative, second_derivative

def gussian_kernel(X, sigma):
  m, n = X.shape
  K = []
  for i in range(m):
    for j in range(m):
      K.append(np.exp(-1.0/sigma*(np.linalg.norm(X[i]-X[j]))**2))
  K = np.reshape(K, (m,m))
  return K

def log_likelihood(beta, K, Y):
  m = Y.shape[0]
  l = 0
  for i in range(m):
    l += -1.0*np.dot(beta.T, K[:,i:i+1]) + np.log(1+np.exp(np.dot(beta.T, K[:,i:i+1].T)))
  return l

if __name__=='__main__':
  csv = ROOT_DIR+'/../../Data/watermelon/watermelon_3.0_x.csv'
  data = np.loadtxt(csv, dtype=float, delimiter=",")
  X = data[1:, 1:-1]
  Y = data[1:, -1]

  ### hyper parameters
  sigma = 0.5
  epsilon = 1e-6

  ### parameters
  alpha = np.random.normal(0, 0.1, (X.shape[0],1))
  b = np.random.normal(0, 0.1)

  beta = np.reshape(np.append(alpha, b), (1,-1))

  K = gussian_kernel(X, sigma)
  K = np.row_stack((K, np.ones((1,K.shape[1]))))
  K = np.column_stack((K, np.ones((K.shape[0],1))))

  m, _ = X.shape
  epoch = 0
  while True:
    beta_epoch = np.zeros(beta.shape)
    for i in range(m):
      first_derivative, second_derivative = log_likelihood_first_second_derivatives(beta, K[:,i:i+1].T, Y[i])
      beta_epoch = beta_epoch + second_derivative/first_derivative
    beta = beta - beta_epoch
    l = log_likelihood(beta, K, Y)
    if epoch%100==0:
      print("Epoch:{} log likelihood:{}".format(epoch, l))
    if l<epsilon:
      break
    epoch += 1
  l = log_likelihood(beta, K, Y)
  print("Epoch:{} log likelihood:{}".format(epoch, l))


