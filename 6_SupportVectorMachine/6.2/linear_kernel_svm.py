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

src = ROOT_DIR+'/../../Data/watermelon/watermelon_3.0_x.csv'
feature_types = [1,1,0]
feature_maker = FeatureMaker(src=src, delimiter=',', types=feature_types, norm=False)
X, Y = feature_maker.make(skip_rows=1, skip_cols=1)
Y = Y.flatten()
print("X:{}\nY:{}".format(X, Y))

problem = svm_problem(Y, X)
params = svm_parameter('-s 0 -t 0 -c 70 -v 8')
model = svm_train(problem, params)
