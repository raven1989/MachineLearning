#! coding:utf-8
## Assuming you had libsvm for python installed.

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import os
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR+"/../Model")
sys.path.append(ROOT_DIR+"/../../0_FeatureMaker")
import numpy as np
from feature_maker import FeatureMaker
from libsvm.svmutil import *

src = ROOT_DIR+'/../../Data/iris/iris_data.csv'
feature_types = [1,1,1,1,0]
feature_maker = FeatureMaker(src=src, delimiter=',', types=feature_types, norm=True)
X, Y = feature_maker.make(skip_rows=1, skip_cols=1)
Y = [len(y)-y.tolist().index(1) for y in Y]
print("X:{}\nY:{}".format(X, Y))

problem = svm_problem(Y, X)
params = svm_parameter('-s 0 -t 2 -c 8 -v 10')
model = svm_train(problem, params)
