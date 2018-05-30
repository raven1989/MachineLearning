import sys
import math
import numpy as np
import collections

class DecisionTreeModel:

  class Feature:
    class FeatureType:
      DISCRETE = 0
      CONTINUOUS = 1
    def __init__(self, fea_name, fea_type, fea_index):
      self.fea_name = fea_name
      self.fea_type = fea_type
      self.fea_index = fea_index
      self.feavalue2index = None
      self.index2feavalue = None 
      if self.fea_type == self.FeatureType.DISCRETE:
        self.feavalue2index = {}
        self.index2feavalue = {}
    @property
    def feature_name(self):
      return self.fea_name
    @property
    def feature_type(self):
      return self.fea_type
    @property
    def feature_index(self):
      return self.fea_index
    def get_index_by_feature_value(self, fea_value):
      if self.fea_type == self.FeatureType.CONTINUOUS:
        raise ValueError("Feature {} has FeatureType.CONTINUOUS, does not support feature value mapping.".format(self.fea_name))
      else:
        return self.feavalue2index.get(fea_value)
    def get_feature_value_by_index(self, fea_idx):
      if self.fea_type == self.FeatureType.CONTINUOUS:
        raise ValueError("Feature {} has FeatureType.CONTINUOUS, does not support feature value mapping.".format(self.fea_name))
      else:
        return self.index2feavalue.get(fea_idx)
    def append_feature_value(self, v):
      if self.feavalue2index.get(v) == None:
        i = len(self.feavalue2index)
        self.feavalue2index[v] = i
        self.index2feavalue[i] = v

  def __init__(self):
    self.data = None
    self.features = None
    self.root = None

  def __data_preprocess__(self, data):
    ## try to reinterpret s to float or int
    def reinterpret_str(s):
      try:
        return eval(s)
      except:
        return s

    if not isinstance(data, np.ndarray):
      raise ValueError("Only support data type {}".format(str(np.ndarray)))
    fea_names = data[0,:]
    values = data[1:,:]
    self.features = []
    m, n = values.shape
    for i, col in enumerate(values.T):
      fea_type = self.Feature.FeatureType.DISCRETE
      if not isinstance(reinterpret_str(col[0]), str):
        fea_type = self.Feature.FeatureType.CONTINUOUS
      feature = self.Feature(fea_names[i], fea_type, i)
      if feature.feature_type == self.Feature.FeatureType.DISCRETE:
        for v in col:
          if feature.get_index_by_feature_value(v) == None:
            feature.append_feature_value(v)
      self.features.append(feature)
    # print(",".join([x.feature_name for x in self.features]))
    # print(",".join([str(x.feature_type) for x in self.features]))
    ## generate new data
    d = [[float(row[i]) if self.features[i].feature_type==self.Feature.FeatureType.CONTINUOUS else self.features[i].get_index_by_feature_value(row[i]) for i in range(len(row))] for row in values]
    self.data = np.array(d)

  class AlgoModel:
    def gain(self, data, feature):
      raise NotImplementedError("Abstract method gain must be implemented.")

  class Id3Model(AlgoModel):
    EPSILON = 1e-6
    def __entropy__(self, labels):
      y = np.reshape(labels, -1)
      y_counter = collections.Counter(y)
      total = sum(y_counter.values())
      ent = 0
      for v in y_counter.values():
        p = v*1.0/total
        ent -= p * math.log(p,2) if p>0.0 else 0.0
      return ent
    ## return gain and boundary value if it's continuous feature
    def gain(self, data, feature):
      # if type(feature) != type(DecisionTreeModel.Feature):
        # raise ValueError("Need type {} for feature".format(DecisionTreeModel.Feature))
      x = data[:,:-1]
      y = data[:,-1]
      total_cnt = len(y)
      whole_ent = self.__entropy__(y)
      d_ent = 0.0
      boundary = 0.0
      gain = 0.0
      if feature.feature_type == DecisionTreeModel.Feature.FeatureType.DISCRETE:
        for fea_idx in feature.index2feavalue:
          d = data[data[:,feature.feature_index]==fea_idx,-1]
          w = len(d)*1.0/total_cnt
          d_ent += self.__entropy__(d) * w
        gain = whole_ent - d_ent
      elif feature.feature_type == DecisionTreeModel.Feature.FeatureType.CONTINUOUS:
        sorted_fea = sorted(np.reshape(data[:,feature.feature_index], -1))
        boundary_candidates = [(sorted_fea[i]+sorted_fea[i+1])/2.0 for i in range(len(sorted_fea)-1)]
        boundary_candidates.append(sorted_fea[0]-self.EPSILON)
        boundary_candidates.append(sorted_fea[-1]+self.EPSILON)
        ## find boundary s.t. min entropy(d,boundary)
        min_b = None
        min_d_ent = None
        for b in boundary_candidates:
          d_lt = data[data[:,feature.feature_index]<=b, -1]
          d_gt = data[data[:,feature.feature_index]>b, -1]
          d_lt_cnt = len(d_lt)
          d_gt_cnt = len(d_gt)
          d_ent = d_lt_cnt*1.0/total_cnt*self.__entropy__(d_lt) \
              + d_gt_cnt*1.0/total_cnt*self.__entropy__(d_gt)
          if min_d_ent == None or min_d_ent > d_ent:
            min_d_ent = d_ent
            min_b = b
        gain = whole_ent - min_d_ent
        boundary = min_b
      return gain, boundary

  ALGO_MODEL_FACTORY = {"id3" : Id3Model}

  def __make_leaf_node__(self, feature, label, samples):
    leaf = {"label":feature.get_feature_value_by_index(label), 
        "is_leaf":True,
        "samples":samples,
        }
    return leaf
  def __make_non_leaf_node__(self, feature, gain, samples, boundary, children):
    non_leaf = {"feature":feature.feature_name, 
        "type":feature.feature_type,
        "is_leaf":False, 
        "gain":gain, 
        "samples":samples, 
        "children":children,
        }
    if non_leaf["type"] == self.Feature.FeatureType.CONTINUOUS:
      non_leaf["boundary"] = boundary,
    return non_leaf

  def __generate_decision_tree_recursively__(self, model, data, feature_candidates):
    node = None
    x = data[:,:-1]
    y = data[:,-1]
    feature_label = self.features[-1]
    is_leaf = False
    samples = len(y)
    label_most, label_most_cnt = collections.Counter(y).most_common(1)[0]
    ## all labels are the same or no feature candidates, it's a leaf node 
    if label_most_cnt == len(y) or len(feature_candidates)==0:
      is_leaf = True
    else:
      ##choose feature s.t. max gain
      fea_gains = []
      max_fea_idx = None
      max_gain = 0.0
      max_boundary = 0.0
      for i,feature in enumerate(feature_candidates):
        fea_gain, boundary = model.gain(data, feature)
        fea_gains.append(fea_gain)
        if fea_gain > max_gain:
          max_fea_idx = i
          max_gain = fea_gain
          max_boundary = boundary
      ## all feature gains are the same, it's a leaf node
      _, gain_most_cnt = collections.Counter(fea_gains).most_common(1)[0]
      if gain_most_cnt == len(fea_gains):
        is_leaf = True
      else:
        chosen_feature = feature_candidates[max_fea_idx]
        children = {}
        next_feature_candidates = [x for x in feature_candidates]
        # print(chosen_feature.feature_name)
        if chosen_feature.feature_type == self.Feature.FeatureType.DISCRETE:
          next_feature_candidates.pop(max_fea_idx)
          for fea_value_idx in chosen_feature.index2feavalue:
            divided = data[data[:,chosen_feature.feature_index]==fea_value_idx,:]
            if len(divided) > 0:
              child = self.__generate_decision_tree_recursively__(model, 
                  divided, next_feature_candidates)
              if child != None:
                children[chosen_feature.get_feature_value_by_index(fea_value_idx)] = child
        elif chosen_feature.feature_type == self.Feature.FeatureType.CONTINUOUS:
          divided_lt = data[data[:,chosen_feature.feature_index]<=max_boundary, :]
          divided_gt = data[data[:,chosen_feature.feature_index]>max_boundary, :]
          child_lt = self.__generate_decision_tree_recursively__(model, 
              divided_lt, next_feature_candidates)
          child_gt = self.__generate_decision_tree_recursively__(model, 
              divided_gt, next_feature_candidates)
          if child_lt != None:
            children[True] = child_lt
          if child_gt != None:
            children[False] = child_gt
        node = self.__make_non_leaf_node__(chosen_feature, max_gain,  
            samples, max_boundary, children)
    if is_leaf == True:
      node = self.__make_leaf_node__(feature_label, label_most, samples)
    return node

  def fit(self, data, algo_model):
    model = self.ALGO_MODEL_FACTORY.get(algo_model)
    if model == None:
      raise ValueError("Algorithm model {} not found, options are {}".fotmat(algo_model, ALGO_MODEL_FACTORY.keys()))
    model = model()
    self.__data_preprocess__(data)
    feature_candidates = [fea for fea in self.features[:-1]]
    self.root = self.__generate_decision_tree_recursively__(model, self.data, feature_candidates)


