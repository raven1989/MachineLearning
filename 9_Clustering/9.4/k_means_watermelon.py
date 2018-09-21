# coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import os
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR+"/../Model")
sys.path.append(ROOT_DIR+"/../../0_FeatureMaker")
from k_means import KMeansCluster
import numpy as np
from feature_maker import FeatureMaker
import json

src = ROOT_DIR+'/../../Data/watermelon/watermelon_4.0.csv'
feature_types = [1,1,0]
feature_maker = FeatureMaker(src=src, delimiter=',', types=feature_types, norm=False)
X, Y = feature_maker.make(skip_rows=1, skip_cols=1, one_hot=False)
# Y = Y.flatten()
print("X:{}".format(X))

### hyper
n_clusters = 3

k_means = KMeansCluster(n_clusters=n_clusters)
k_means.fit(X)

