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
feature_types = [1,1]
feature_maker = FeatureMaker(src=src, delimiter=',', types=feature_types, norm=False)
X, Y = feature_maker.make(skip_rows=1, skip_cols=1)
Y = Y.flatten()
print("X:{}\nY:{}".format(X, Y))

# sys.exit(0)
problem = svm_problem(Y, X)
### 样本数16条，很少，高斯核和线性核不会相差太多，p epsilon-损失不敏感函数的epsilon
params = svm_parameter('-s 3 -t 0 -b 1 -p 0.01 -v 8')
# params = svm_parameter('-s 3 -t 2 -b 1 -p 0.01 -v 8')
model = svm_train(problem, params)
