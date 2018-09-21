# coding:utf-8
import numpy as np

class KMeansCluster:
  
  def __init__(self, n_clusters):
    self.n_clusters = n_clusters
    self.cluster_centers = None

  def fit(self, X):
    self.__init_centers__(X)
    print("Init Centers:{}".format(self.cluster_centers))
    is_repeat = True
    epoch = 0
    epsilon = 0.0
    while is_repeat:
      ### loss是最小平方误差 : E = sigma_i sigma_x ||x-mu_i||**2
      ### mu_i是第i个簇的均值向量
      loss = 0.0
      is_repeat = False
      C = []
      for x in X:
        dist = np.linalg.norm(x-self.cluster_centers, axis=1)
        c = np.argmin(dist)
        # print("dist:{} c:{}".format(c, dist))
        loss += dist[c]**2
        C.append(c)
      print("Epoch:{} loss:{} C:{}".format(epoch, loss, C))
      C = np.array(C)
      for c in range(self.n_clusters):
        mu = np.mean(X[C==c], axis=0)
        # print("centers:{} update:{}".format(self.cluster_centers[c], mu))
        if np.linalg.norm(mu-self.cluster_centers[c])>epsilon:
          self.cluster_centers[c] = mu
          is_repeat = True
      epoch += 1
      # break

  def __init_centers__(self, X):
    m, _ = X.shape
    indice = np.random.choice(a=m, size=self.n_clusters, replace=False)
    # print("indice:{}".format(indice))
    self.cluster_centers = X[indice,:]
