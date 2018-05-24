import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def GetProjectivePoint_2D(point, line):
  a = point[0]
  b = point[1]
  k = line[0]
  t = line[1]

  if   k == 0:      return [a, t]
  elif k == np.inf: return [0, b]
  x = (a+k*b-k*t) / (k*k+1)
  y = k*x + t
  return [x, y]

if __name__ == "__main__":
  ## load data from csv
  data = np.loadtxt("water_melon.csv", dtype=float, delimiter=",")
  x = data[1:,1:-1]
  y = data[1:,-1]
  pos_data = x[np.where(y==1)]
  neg_data = x[np.where(y==0)]
  # print(pos_data)
  # print(neg_data)

  f1= plt.figure(1)
  
  # ## sklearn lda model
  # lda_model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None).fit(x, y)

  # ## sklearn lda model
  # h = 0.001
  # x0, x1 = np.meshgrid(np.arange(-1, 1, h), np.arange(-1, 1, h))
  # z = lda_model.predict(np.c_[x0.ravel(), x1.ravel()])
  # print(z)
  # z = z.reshape(x0.shape)
  # plt.contourf(x0, x1, z)

  ## draw scatter diagram
  plt.title("watermelon.csv")
  plt.xlabel("density")
  plt.ylabel("sugar_ratio")
  plt.scatter(x[y==0,0], x[y==0,1], marker='x', color='k', s=100, label='neg')
  plt.scatter(x[y==1,0], x[y==1,1], marker='o', color='g', s=100, label='pos')
  plt.legend(loc='upper right')

  plt.xlim( -0.2, 1 )
  plt.ylim( -0.5, 0.7 )

  ## mean
  pos_mean = np.mean(pos_data, axis=0)
  neg_mean = np.mean(neg_data, axis=0)
  # print(pos_mean, neg_mean)
  u = neg_mean-pos_mean
  # print(u)

  # m,n = np.shape(x)
  # Sw = np.zeros((n,n))
  # for i in range(m):
      # x_tmp = x[i].reshape(n,1)  # row -> cloumn vector
      # if y[i] == 0: u_tmp = neg_mean.reshape(n,1)
      # if y[i] == 1: u_tmp = pos_mean.reshape(n,1)
      # Sw += np.dot( x_tmp - u_tmp, (x_tmp - u_tmp).T )
  # # print(Sw)

  # ## svd
  # U, sigma, Vt = np.linalg.svd(Sw)
  # Sw_inv = Vt.T * np.linalg.inv(np.diag(sigma)) * U.T

  # w = np.dot(Sw_inv, u.T)
  # # print(w)

  # ## draw the projection line 
  # k = -w[0]/w[1]
  # p0_x0 = x[:,0].max()
  # p0_x1 = k*p0_x0
  # p1_x0 = -p0_x0
  # p1_x1 = k*p1_x0
  # plt.plot([p0_x0,p0_x1], [p1_x0,p1_x1])

  ## cov
  pos_cov = np.cov(pos_data.T)
  neg_cov = np.cov(neg_data.T)
  # print(pos_cov, neg_cov)
  Sw = pos_cov + neg_cov
  # print(Sw)

  ## svd
  U, sigma, Vt = np.linalg.svd(Sw)
  Sw_inv = Vt.T * np.linalg.inv(np.diag(sigma)) * U.T

  w = np.dot(Sw_inv, u.T)
  # print(w)

  ## draw the projection line 
  k = w[1]/w[0]
  p0_x0 = x[:,0].max()
  p0_x1 = k*p0_x0
  p1_x0 = -p0_x0
  p1_x1 = k*p1_x0
  plt.plot([p0_x0,p1_x0], [p0_x1,p1_x1])

  ## predict
  # print(np.dot(x, w))

  ## drew projections
  m,n = np.shape(x)
  for i in range(m):
    x_p = GetProjectivePoint_2D( [x[i,0],x[i,1]], [k,0] ) 
    if y[i] == 0: 
        plt.plot(x_p[0], x_p[1], 'kx', markersize = 5)
    if y[i] == 1: 
        plt.plot(x_p[0], x_p[1], 'go', markersize = 5)   
    plt.plot([ x_p[0], x[i,0]], [x_p[1], x[i,1] ], 'c--', linewidth = 0.3)

  ## drew means' projections
  pos_mean_projection = GetProjectivePoint_2D(pos_mean, [k,0])
  neg_mean_projection = GetProjectivePoint_2D(neg_mean, [k,0])
  plt.plot(pos_mean_projection[0], pos_mean_projection[1], 'ro', markersize=8)
  plt.plot(neg_mean_projection[0], neg_mean_projection[1], 'rx', markersize=8)

  plt.show()
