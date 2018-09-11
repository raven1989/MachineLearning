#! coding:utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf8')
import os
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR+"/../Model")
sys.path.append(ROOT_DIR+"/../../0_FeatureMaker")
import numpy as np
from feature_maker import FeatureMaker
import json

src = ROOT_DIR+'/../../Data/watermelon/watermelon_3.0.csv'
feature_types = [0,0,0,0,0,0,1,1,0]
feature_maker = FeatureMaker(src=src, delimiter=',', types=feature_types, norm=False)
X, Y = feature_maker.make(skip_rows=1, skip_cols=1, one_hot=False)
Y = Y.flatten()
print("X:{}\nY:{}".format(X, Y))

class_prior = {}
class_cond = {}
x_feature_types = feature_types[:-1]

### train
for x,y in zip(X, Y):
  # print("x:{} y:{}".format(x, y))
  if class_prior.get("total") is None:
    class_prior["total"] = 0
  class_prior["total"] += 1
  if class_prior.get(y) is None:
    class_prior[y] = 0
    class_cond[y] = [{} for i in range(len(x_feature_types))]
  class_prior[y] += 1
  for i,f in enumerate(x):
    if x_feature_types[i]==0:
      if class_cond[y][i].get(f) is None:
        class_cond[y][i][f] = 0
      ### 计数
      class_cond[y][i][f] += 1
    ### 如果是连续特征，累加值
    elif x_feature_types[i]==1:
      if class_cond[y][i].get("sum") is None:
        class_cond[y][i]["sum"] = 0.0
      class_cond[y][i]["sum"] += f

print(json.dumps(class_prior, indent=4))
print(json.dumps(class_cond, indent=4))

for x,y in zip(X, Y):
  ### 当前类取值为y的个数
  y_total_cnt = class_prior[y]
  for i,f in enumerate(x):
    if x_feature_types[i]==0:
      ### 当前特征取值为f的个数
      f_cnt = class_cond[y][i].get(f, 0)
      if f_cnt>=1 or f_cnt==0:
          ### 当前特征可取值的个数
          class_f_cnt = len(feature_maker.feature2index[i])
          ### 拉普拉斯平滑
          cond_prob = 1.0*(f_cnt+1) / (y_total_cnt+class_f_cnt)
          print("y:{} feature:{} name:{} index:{} cond prob:{}".format(y, i, feature_maker.index2feature[i].get(f), f, cond_prob))
          class_cond[y][i][f] = np.log(cond_prob)
    elif x_feature_types[i]==1:
      ### 计算平均值和方差
      mean = class_cond[y][i]["sum"]*1.0/y_total_cnt
      class_cond[y][i]["mean"] = mean
      square_variance = np.square(f-mean)
      if class_cond[y][i].get("variance") is None:
        class_cond[y][i]["variance"] = 0
      class_cond[y][i]["variance"] += square_variance*1.0/y_total_cnt


### test
test = ["青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.697, 0.460]
# test = ["浅白", "硬挺", "浊响", "清晰", "凹陷", "硬滑", 0.697, 0.460]
test = [feature_maker.feature2index[i].get(test[i]) if x_feature_types[i]==0 else test[i] for i in range(len(test))]
print(test)
total_cnt = class_prior.get("total")
### y取值的个数
class_y_cnt = len(feature_maker.feature2index[-1])
p = [0] * class_y_cnt 
for y in feature_maker.feature2index[-1].values():
  ### 当前类取值为y的个数
  y_total_cnt = class_prior.get(y, 0)
  prior_prob = (y_total_cnt+1)*1.0 / (total_cnt+class_y_cnt)
  log_prior_prob = np.log(prior_prob)
  p[int(y)] += log_prior_prob
  for i,f in enumerate(test):
    ### 当前特征可取值的个数
    class_f_cnt = len(feature_maker.feature2index[i])
    if x_feature_types[i]==0:
      default_cond_prob = 1.0/(y_total_cnt+class_f_cnt)
      log_cond_prob = class_cond[y][i].get(f, default_cond_prob)
    elif x_feature_types[i]==1:
      mean = class_cond[y][i].get("mean")
      variance = class_cond[y][i].get("variance")
      cond_prob = 1.0/np.sqrt(2.0*np.pi*variance)*np.exp(-0.5*np.square(f-mean)/variance)
      log_cond_prob = np.log(cond_prob)
    else:
      print("Unexpected feature type:{}".format(x_feature_type[i]))
    print("log cond prop:{}".format(cond_prob))
    p[int(y)] += log_cond_prob

print(p)
