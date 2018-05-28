#coding:utf-8
import sys
sys.path.append("../tools")
import math
import numpy as np
# from functools import reduce
import collections
from treetoolkit import DNode, plot_dtree

def reinterpret_str(s):
  try:
    return eval(s)
  except:
    return s

## preprocess data
## remain int float
## mapping str to index
def data_preprocess(data):
  m, n = data.shape
  fea2index = []
  index2fea = []
  fea_type = []
  for all_fea in data.T:
    i = 0
    fea_map = {}
    index_map = {}
    is_discrete = False
    for fea in all_fea:
      if isinstance(reinterpret_str(fea),str) and fea_map.get(fea)==None:
        is_discrete = True
        fea_map[fea] = i
        index_map[i] = fea
        i += 1
    fea2index.append(fea_map)
    index2fea.append(index_map)
    fea_type.append("discrete" if is_discrete else "continuos")
  d = [[float(row[i]) if fea2index[i].get(row[i])==None else fea2index[i][row[i]] for i in range(len(row))] for row in data]
  return np.array(d), fea2index, index2fea, fea_type

## compute entropy based on labels as 4.1
def entropy(label):
  y2cnt = {}
  y = np.reshape(label, -1)
  total = 0
  for label in y:
    if y2cnt.get(label)==None:
      y2cnt[label] = 1 
    else:
      y2cnt[label] += 1
    total += 1
  ent = 0
  for c in y2cnt.values():
    p = c*1.0/total
    ent -= p*math.log(p,2) if p>0.0 else 0.0
    # print(c, p)
  return ent

## compute Gain(D,a) for discrete features as 4.2
## feature_idx indicates the feature a
def gain_discrete(data, feature_idx, ent_label=None):
  if ent_label==None:
    ent_label = entropy(data[:,-1])
  feature2values = {}
  total = 0
  for fea in np.reshape(data[:,feature_idx],-1):
    if feature2values.get(fea)==None:
      feature2values[fea] = 1
    else:
      feature2values[fea] += 1
    total += 1
  # print(feature2values, total)
  d_ent = 0
  for fea,v in feature2values.items():
    d_data = data[data[:,feature_idx]==fea, -1]
    # print(d_data)
    v_ent = entropy(d_data)
    w = feature2values[fea]*1.0/total
    # print(w, v_ent)
    d_ent += w*v_ent
  return ent_label-d_ent

## compute Gain(D,a) for continuous features as 4.8
## feature_idx indicates the feature a
## return gain and classification boundary value
def gain_continuous(data, feature_idx, ent_label=None):
  if ent_label==None:
    ent_label = entropy(data[:,-1])
  f = sorted(np.reshape(data[:,feature_idx],-1))
  # print(f)
  t = [(f[i]+f[i+1])/2.0 for i in range(len(f)-1)]
  epsilon = 1e-6
  t.append(f[0]-epsilon)
  t.append(f[-1]+epsilon)
  # print(t)
  ## find t s.t. min entropy(dt)
  min_t = None
  min_t_ent = None
  for it in t:
    t1 = data[data[:,feature_idx]<=it, -1]
    t2 = data[data[:,feature_idx]>it, -1]
    num_t1 = len(t1)
    num_t2 = len(t2)
    num_t = (num_t1+num_t2)*1.0
    ent_t = num_t1/num_t*entropy(t1) + num_t2/num_t*entropy(t2)
    # print(t1, t2, num_t1/num_t*entropy(t1), num_t2/num_t*entropy(t2), num_t1, num_t2, num_t, ent_t)
    if min_t_ent==None or min_t_ent>ent_t:
      min_t_ent = ent_t
      min_t = it
  # print(min_t_ent, min_t)
  # ## t is empty, means only one sample in data, now entropy is 0
  # if min_t_ent==None:
  #   min_t_ent = 0.0
  return ent_label-min_t_ent, min_t

def gen_decision_tree_recursively(data, fea_idx_list, fea2index, index2fea, fea_name, fea_type):
  x = data[:,:-1]
  y = data[:,-1]
  whole_ent = entropy(y) 
  ## y_ent==0 means all ys' labels are the same
  y_ent = whole_ent
  is_leaf = False
  samples = len(y)
  gain = 0.0
  value = "default"
  chosen_fea_idx = 0
  most_label = collections.Counter(y).most_common(1)[0][0]
  if not y_ent<0 and not y_ent>0:
    value = index2fea[-1].get(most_label)
    is_leaf = True
  ## fea_idx_list is empty or gains are the same on all features, then it's a leaf node labeled on the most ys
  if len(fea_idx_list)<=0: 
    value = index2fea[-1].get(most_label)
    is_leaf = True
  else:
    gains_collection = []
    max_gain = 0.0
    max_fea_idx = 0
    max_boundary = 0.0
    for fea_idx in fea_idx_list:
      if fea_type[fea_idx]=="discrete":
        fea_gain = gain_discrete(data, fea_idx, whole_ent)
        gains_collection.append("%.6f"%fea_gain)
        if fea_gain>max_gain:
          max_gain = fea_gain
          max_fea_idx = fea_idx
      else:
        fea_gain, boundary = gain_continuous(data, fea_idx, whole_ent)
        gains_collection.append("%.6f"%fea_gain)
        if fea_gain>max_gain:
          max_gain = fea_gain
          max_fea_idx = fea_idx
          max_boundary = boundary
    ## if gains are all the same, then it's a leaf node
    # print("fea_name:{} gain:{} boundary:{}".format(fea_name[max_fea_idx], max_gain, max_boundary))
    if len(collections.Counter(gains_collection))<=1:
      value = index2fea[-1].get(most_label)
      is_leaf = True
    ## else choose max_gain's fea to build a non-leaf node
    else:
      gain = max_gain
      chosen_fea_idx = max_fea_idx
      value = max_boundary
      is_leaf = False
  if is_leaf:
    leaf_node = DNode(samples)
    leaf_node.set_leaf(label=value)
    return leaf_node
  else:
    chosen_fea_type = fea_type[chosen_fea_idx]
    dnode = DNode(samples=samples, 
        gain=gain, 
        fea_name=fea_name[chosen_fea_idx], 
        fea_type=chosen_fea_type,
        fea_value=value)
    ## recursively build tree
    next_fea_idx_list = fea_idx_list  
    if chosen_fea_type=="discrete":
      ## remove chosen_fea_idx from fea_idx_list if it's discrete feature
      next_fea_idx_list = filter(lambda k:k!=chosen_fea_idx, fea_idx_list)
      for fea_value in index2fea[chosen_fea_idx].keys():
        next_data = data[data[:,chosen_fea_idx]==fea_value,:]
        if len(next_data)>0:
          child = gen_decision_tree_recursively(next_data, next_fea_idx_list, fea2index, index2fea, fea_name, fea_type)
          if child!=None:
            dnode.append_child(child, index2fea[chosen_fea_idx].get(fea_value,"Unknown")) 
    else:
      epsilon = 1e-6
      fea_value = value
      next_data_1 = data[data[:,chosen_fea_idx]<=fea_value,:]
      next_data_2 = data[data[:,chosen_fea_idx]>fea_value,:]
      child1 = gen_decision_tree_recursively(next_data_1, next_fea_idx_list, fea2index, index2fea, fea_name, fea_type)
      child2 = gen_decision_tree_recursively(next_data_2, next_fea_idx_list, fea2index, index2fea, fea_name, fea_type)
      if child1!=None:
        dnode.append_child(child1, fea_value-epsilon)
      if child2!=None:
        dnode.append_child(child2, fea_value+epsilon)
    return dnode

def display(x):
  print(x)
  print("-------------")

if __name__=='__main__':
  data = np.loadtxt("../watermelon/watermelon.csv", dtype=str, delimiter=",", skiprows=0)
  fea_name = data[0,:]
  data = data[1:,:]
  # print(data)
  data, fea2index, index2fea, fea_type = data_preprocess(data)
  # print(data, fea2index, index2fea, fea_type)
  # print(index2fea[-1].get(0))
  whole_ent = entropy(data[:,-1])
  # print(whole_ent)
  # gain_1 = gain_discrete(data, 1, ent_label=whole_ent)
  # print(gain_1)
  # gain_2, boundary = gain_continuous(data, 7, ent_label=whole_ent)
  # print(gain_2, boundary)
  fea_idx_list = [i for i in range(1, len(fea_name)-1)]
  # print(fea_name, fea_idx_list)
  root = gen_decision_tree_recursively(data=data, 
      fea_idx_list=fea_idx_list, 
      fea2index=fea2index, 
      index2fea=index2fea, 
      fea_name=fea_name, 
      fea_type=fea_type)
  # print(root)
  # print("-----------------------")
  # print(root.children)
  # print(root.next_node("清晰"))
  # print("-----------------------")
  # # print(root.next_node("清晰").children)
  # # print("-----------------------")
  # print(root.next_node("清晰").next_node(0.01))
  # print("-----------------------")
  # print(root.next_node("清晰").next_node(0.5))
  # print("-----------------------")
  # print(root.next_node("稍糊"))
  # print("-----------------------")
  # print(root.next_node("模糊"))
  # root.traverse_pre_order_recursively(do=display)
  plot_dtree(root)

