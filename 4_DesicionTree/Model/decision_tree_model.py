import sys
import math
import numpy as np
import collections
from graphviz import Digraph
import seaborn as sns

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
        # print("{} : {}".format(feature.feature_name, feature.feavalue2index))
      self.features.append(feature)
    # print(",".join([x.feature_name for x in self.features]))
    # print(",".join([str(x.feature_type) for x in self.features]))
    ## generate new data
    d = [[float(row[i]) if self.features[i].feature_type==self.Feature.FeatureType.CONTINUOUS else self.features[i].get_index_by_feature_value(row[i]) for i in range(len(row))] for row in values]
    self.data = np.array(d)

  class AlgoModel:
    @property
    def name(self):
      return "Abstract Model"
    def select_feature(self, data, feature_candidates):
      raise NotImplementedError("Abstract method gain must be implemented.")

  class Id3Model(AlgoModel):
    EPSILON = 1e-6
    @property
    def name(self):
      return "Entropy Gain"
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
    def __gain__(self, data, feature):
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
        # boundary_candidates.append(sorted_fea[0]-self.EPSILON)
        # boundary_candidates.append(sorted_fea[-1]+self.EPSILON)
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
    ## choose feature s.t. max gain
    def select_feature(self, data, feature_candidates):
      is_leaf = False
      fea_gains = []
      max_fea_idx = None
      max_gain = 0.0
      max_boundary = 0.0
      for i,feature in enumerate(feature_candidates):
        fea_gain, boundary = self.__gain__(data, feature)
        fea_gains.append(fea_gain)
        if fea_gain > max_gain:
          max_fea_idx = i
          max_gain = fea_gain
          max_boundary = boundary
      ## all feature gains are the same, it's a leaf node
      _, gain_most_cnt = collections.Counter(fea_gains).most_common(1)[0]     
      if gain_most_cnt == len(fea_gains):
        is_leaf = True
      return is_leaf, max_fea_idx, max_gain, max_boundary

  class CartModel(AlgoModel):
    @property
    def name(self):
      return "Gini"
    def __gini__(self, labels):
      y = np.reshape(labels, -1)
      y_counter = collections.Counter(y)
      total = sum(y_counter.values())
      gini = 1.0
      for n in y_counter.values():
        pk = n*1.0/total
        gini -= pk * pk
      return gini
    ## compute feature's gini index
    def __gini_index__(self, data, feature):
      gini_index = None
      boundary = 0.0
      total = len(data[:,-1])
      if feature.feature_type == DecisionTreeModel.Feature.FeatureType.DISCRETE:
        gini_index = 0.0
        for fea_idx in feature.index2feavalue:
          divided = data[data[:,feature.feature_index]==fea_idx, -1]
          n_samples = len(divided)
          w = n_samples*1.0/total
          gini_index += w * self.__gini__(divided)
      elif feature.feature_type == DecisionTreeModel.Feature.FeatureType.CONTINUOUS:
        sorted_fea = sorted(np.reshape(data[:,feature.feature_index], -1))
        boundary_candidates = [sorted_fea[i]-(sorted_fea[i]-sorted_fea[i+1])/2.0 
            for i in range(len(sorted_fea)-1)]
        gini_index = None
        boundary = None
        for b in boundary_candidates:
          label_lt = data[data[:,feature.feature_index]<=b, -1]
          label_gt = data[data[:,feature.feature_index]>b, -1]
          w_lt = len(label_lt)*1.0/total
          w_gt = len(label_gt)*1.0/total
          cur_gini_index = self.__gini__(label_lt)*w_lt + self.__gini__(label_gt)*w_gt
          if gini_index == None or cur_gini_index < gini_index:
            gini_index = cur_gini_index
            boundary = b
      return gini_index, boundary
    ## choose feature s.t. min gini_index(d,f)
    def select_feature(self, data, feature_candidates):
      is_leaf = False
      gini_indexes = []
      chosen_gini_index = None
      chosen_feature_idx = 0
      chosen_boundary = 0.0
      for i,feature in enumerate(feature_candidates):
        fea_idx = feature.feature_index
        gini_index, boundary = self.__gini_index__(data, feature)
        gini_indexes.append(gini_index)
        if chosen_gini_index == None or gini_index < chosen_gini_index:
          chosen_gini_index = gini_index
          chosen_feature_idx = i
          chosen_boundary = boundary
      gini_counter = collections.Counter(gini_indexes)
      if gini_counter.most_common(1)[0] == len(gini_indexes):
        is_leaf = True
      return is_leaf, chosen_feature_idx, chosen_gini_index, chosen_boundary

  ALGO_MODEL_FACTORY = {"id3" : Id3Model, 
      "cart" : CartModel}

  def __make_leaf_node__(self, feature, label, samples):
    leaf = {"label":feature.get_feature_value_by_index(label), 
        "feature_index":feature.feature_index,
        "is_leaf":True,
        "samples":samples,
        }
    return leaf
  def __make_non_leaf_node__(self, model, feature, model_value, samples, boundary, children):
    non_leaf = {"feature":feature.feature_name, 
        "type":feature.feature_type,
        "feature_index":feature.feature_index,
        "is_leaf":False, 
        model.name:"%.4f"%float(model_value), 
        "samples":samples, 
        "children":children,
        }
    if non_leaf["type"] == self.Feature.FeatureType.CONTINUOUS:
      non_leaf["boundary"] = boundary
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
      ## choose a feature to divide data
      is_leaf, chosen_fea_index, chosen_value, chosen_boundary = model.select_feature(data, feature_candidates)
      if not is_leaf: 
        chosen_feature = feature_candidates[chosen_fea_index]
        children = {}
        next_feature_candidates = [x for x in feature_candidates]
        # print(chosen_feature.feature_name)
        if chosen_feature.feature_type == self.Feature.FeatureType.DISCRETE:
          next_feature_candidates.pop(chosen_fea_index)
          for fea_value_idx in chosen_feature.index2feavalue:
            divided = data[data[:,chosen_feature.feature_index]==fea_value_idx,:]
            if len(divided) > 0:
              child = self.__generate_decision_tree_recursively__(model, 
                  divided, next_feature_candidates)
              if child != None:
                children[chosen_feature.get_feature_value_by_index(fea_value_idx)] = child
            ## if divided is empty, add a leaf node labeled the most label of before-divided y as a leaf child
            ## when feature is continuous, divided data has at least 1 sample, so this situation will never occur.
            else:
              children[chosen_feature.get_feature_value_by_index(fea_value_idx)] = \
                  self.__make_leaf_node__(feature_label, label_most, 0)
        elif chosen_feature.feature_type == self.Feature.FeatureType.CONTINUOUS:
          divided_lt = data[data[:,chosen_feature.feature_index]<=chosen_boundary, :]
          divided_gt = data[data[:,chosen_feature.feature_index]>chosen_boundary, :]
          child_lt = self.__generate_decision_tree_recursively__(model, 
              divided_lt, next_feature_candidates)
          child_gt = self.__generate_decision_tree_recursively__(model, 
              divided_gt, next_feature_candidates)
          if child_lt != None:
            children[True] = child_lt
          if child_gt != None:
            children[False] = child_gt
          # print(chosen_boundary)
        node = self.__make_non_leaf_node__(model, chosen_feature, chosen_value,  
            samples, chosen_boundary, children)
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

  def predict(self, data):
    labels = []
    for sample in data:
      cursor = self.root
      if cursor == None:
        raise ValueError("DecisionTreeModel is not ready, call fit first.")
      children = cursor.get("children")
      while children!=None:
        fea_name = cursor.get("feature")
        fea_type = cursor.get("type")
        fea_idx = cursor.get("feature_index")
        fea = self.features[fea_idx]
        if fea_type == self.Feature.FeatureType.DISCRETE:
          key = sample[fea_idx]
        elif fea_type == self.Feature.FeatureType.CONTINUOUS:
          boundary = cursor.get("boundary")
          key = sample[fea_idx]<=boundary
        else:
          raise ValueError("DecisionTreeModel feature does not has type:{}".format(fea_type))
        cursor = children.get(key)
        if cursor == None:
          raise ValueError("DecisionTreeModel feature:{} does not recognize value:{}".format(fea_name, sample[fea_idx]))
        children = cursor.get("children")
      label = cursor.get("label")
      labels.append(label)
    return np.array(labels)

  def __node_str__(self, node):
    show = []
    for k in node:
      if k not in ["children","type","is_leaf","feature_index"]:
        show.append(str(k)+":"+str(node[k]).decode('utf-8'))
    return "\n".join(show)

  def __color_brew__(self, n):
    """Generate n colors with equally spaced hues.

      Parameters
      ----------
      n : int
          The number of colors required.

      Returns
      -------
      color_list : list, length n
          List of n tuples of form (R, G, B) being the components of each color.
      """
    color_list = []

    # Initialize saturation & value; calculate chroma & value shift
    s, v = 0.75, 0.9
    c = s * v
    m = v - c

    for h in np.arange(25, 385, 360. / n).astype(int):
      # Calculate some intermediate values
        h_bar = h / 60.
        x = c * (1 - abs((h_bar % 2) - 1))
        # Initialize RGB with same hue & chroma as our color
        rgb = [(c, x, 0),
            (x, c, 0),
            (0, c, x),
            (0, x, c),
            (x, 0, c),
            (c, 0, x),
            (c, x, 0)]
        r, g, b = rgb[int(h_bar)]
        # Shift the initial RGB values to match value and store
        rgb = [(int(255 * (r + m))),
            (int(255 * (g + m))),
            (int(255 * (b + m)))]
        color_list.append(rgb)

    return color_list

  def __color_brew_from_seaborn__(self):
    # palette = sns.color_palette("BuGn_r").as_hex()
    # palette = sns.color_palette("husl", 8).as_hex()
    palette = sns.color_palette("GnBu_d").as_hex()
    return palette

  def __get_color__(self, color):
    # Return html color code in #RRGGBBAA format
    # alpha = 0
    # color.append(alpha)
    hex_codes = [str(i) for i in range(10)]
    hex_codes.extend(['a', 'b', 'c', 'd', 'e', 'f'])
    color = [hex_codes[c // 16] + hex_codes[c % 16] for c in color]
    return '#' + ''.join(color)

  def export_graphviz(self):
    if self.root == None:
      raise ValueError("DecisionTreeModel is not ready, call fit first.")
    n_features = len(self.features)
    # color_list = self.__color_brew__(n_features)
    color_list = self.__color_brew_from_seaborn__()
    # print(color_list)
    # dot = Digraph(comment = "Decision Tree", node_attr={"shape":"component"})
    # dot = Digraph(comment = "Decision Tree", node_attr={"shape":"box", "style":"rounded,filled"})
    dot = Digraph(comment = "Decision Tree", node_attr={"shape":"note", "style":"rounded,filled"})
    cnt = 0
    q = [(str(cnt), self.root)]
    for_show = []
    label = self.__node_str__(self.root)
    # color = "%d,%d,%d" % tuple(color_list[self.root.get("feature_index",0)%len(color_list)])
    # color = self.__get_color__(color_list[self.root.get("feature_index",0)%len(color_list)])
    color = color_list[self.root.get("feature_index",0)%len(color_list)]
    # print(color)
    dot.node(name=str(cnt), label=label, fillcolor=color)
    parent_name = str(cnt)
    cnt += 1
    while len(q) > 0:
      parent_name, parent = q.pop(0)
      children = parent.get("children")
      if children!=None and len(children)>0:
        for k,v in children.items():
          name = str(cnt)
          # color = "%d,%d,%d" % tuple(color_list[v.get("feature_index",0)%len(color_list)])
          # color = self.__get_color__(color_list[v.get("feature_index",0)%len(color_list)])
          color = color_list[v.get("feature_index",0)%len(color_list)]
          # print(color)
          dot.node(name=name, label=self.__node_str__(v), fillcolor=color)
          dot.edge(parent_name, name, label=str(k))
          cnt += 1
          q.append((name,v))
    return dot


