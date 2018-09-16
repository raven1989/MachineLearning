import sys
import numpy as np
import hashlib
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import bisect

import json

class FeatureMaker:
  '''Feature Maker Class

  Parameters
  ----------
  src : str
    data src, eg:file name.
  delimiter : str
    delimiter of data src, eg:",".
  types : list
    types of columns from data src; 0:discrete feature; 1:continuous feature. eg:[0,0,1]
  norm : bool
    whether to normalize continuous features.

  Returns
  -------
  x : ndarray
  y : ndarray
  '''
  def __init__(self, src, delimiter, types, norm):
    self.src = src
    self.delimiter = delimiter
    self.types = types
    self.norm = norm
    ## exclude last col which is y
    discrete_feature_idx = filter(lambda i:self.types[i]==0, range(len(self.types)-1))
    self.one_hot_encoder = OneHotEncoder(categorical_features=discrete_feature_idx, sparse=False)
    self.label_one_hot_encoder = None
    if self.norm:
      self.min_max_scaler = MinMaxScaler()
    self.feature2index = None
    self.index2feature = None

  def make(self, skip_rows=0, skip_cols=0, one_hot=True):
    raw_data = []
    self.feature2index = [{} for t in self.types]
    self.index2feature = [{} for t in self.types]
    with open(self.src) as src:
      for row,line in enumerate(src):
        # print(row, line)
        if row<skip_rows:
          continue
        splitted = line.strip().split(self.delimiter)[skip_cols:]
        raw_sample = []
        for i,t in enumerate(self.types):
          if t==0:
            value = self.feature2index[i].get(splitted[i])
            if value is None:
              value = float(len(self.feature2index[i]))
              self.feature2index[i][splitted[i]] = value
              self.index2feature[i][value] = splitted[i]
          else:
            value = float(splitted[i])
          raw_sample.append(value)
        raw_data.append(raw_sample)
    raw_data = np.array(raw_data)
    x = raw_data[:,:-1]
    y = raw_data[:,-1:]
    if not one_hot:
      return x, y
    # print(raw_data)
    x = self.one_hot_encoder.fit_transform(X=x, y=y)#.toarray()
    if self.norm:
      x = self.min_max_scaler.fit_transform(X=x, y=y)
    ## one-hot encoding y if needed
    if len(self.feature2index[-1])>2:
      self.label_one_hot_encoder = OneHotEncoder(categorical_features=[0], sparse=False)
      y = self.label_one_hot_encoder.fit_transform(X=y)
    return x, y

  def shuffle(self, x, y):
    return shuffle(x, y)

  def train_test_split(self, x, y, test_size=0.1, train_size=None, random_state=None, shuffle=True, stratify=None):
    '''
    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.

    test_size : float, int, None, optional
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. By default, the value is set to 0.25.
        The default will change in version 0.21. It will remain 0.25 only
        if ``train_size`` is unspecified, otherwise it will complement
        the specified ``train_size``.

    train_size : float, int, or None, default None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    shuffle : boolean, optional (default=True)
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.

    stratify : array-like or None (default is None)
        If not None, data is split in a stratified fashion, using this as
        the class labels.
    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.

        .. versionadded:: 0.16
            If the input is sparse, the output will be a
            ``scipy.sparse.csr_matrix``. Else, output type is the same as the
            input type.
    '''
    return train_test_split(x, y, test_size=test_size, train_size=train_size, random_state=random_state, shuffle=True, stratify=stratify)

  def encode(self, X, Y=None, one_hot=True):
    ## exclude last col which is y
    discrete_feature_idx = filter(lambda i:self.types[i]==0, range(len(self.types)-1))
    x = np.array([[self.feature2index[i].get(f, f) for i,f in enumerate(xx)] for xx in X])
    y = np.array([[self.feature2index[-1].get(f, f) for f in yy] for yy in Y]) if Y is not None else Y
    # print("x:{}".format(x))
    if one_hot:
      x = self.one_hot_encoder.transform(x)
      if y is not None and self.label_one_hot_encoder is not None:
        y = self.label_one_hot_encoder.transform(y) 
    if self.norm:
      x = self.min_max_scaler.transform(X=x, y=y)
    if y is not None:
      return x, y
    return x

  def decode(self, X, Y=None, one_hot=True):
    x = X
    y = Y
    ## exclude last col which is y
    discrete_feature_idx = filter(lambda i:self.types[i]==0, range(len(self.types)-1))
    continuous_feature_num = len(self.types)-1-len(discrete_feature_idx)
    if one_hot:
      x = []
      for xx in X:
        discrete_feature_end = -continuous_feature_num
        indice = filter(lambda i:xx[i]==1, range(len(xx[:discrete_feature_end])))
        discrete_x = self.one_hot_encoder.active_features_[indice] - self.one_hot_encoder.feature_indices_[:-1]
        discrete_x = discrete_x.tolist()
        continuous_x = xx[discrete_feature_end:].tolist()
        # print("discrete:{}".format(type(discrete_x)))
        # print("continuous:{}".format(continuous_x))
        # print(discrete_x+continuous_x)
        x.append(discrete_x+continuous_x)
      if y is not None and self.label_one_hot_encoder is not None:
        y = []
        for yy in Y:
          indice = filter(lambda i:yy[i]==1, range(len(yy)))
          y = self.label_one_hot_encoder.active_features_[indice]
    x = [[self.index2feature[i].get(f) if self.types[i]==0 else f for i,f in enumerate(xx)] for xx in x]
    if y is not None:
      y = [[self.index2feature[-1].get(f) for f in yy] for yy in y]
      return x, y
    return x

  def get_property_one_hot_encoded_index(self, feature, prop):
    if self.one_hot_encoder is None:
      return False
    idx = int(self.feature2index[feature].get(prop))
    if feature==-1 or feature+1==len(self.feature2index):
      offset = 0
    elif feature>=0 and feature<len(self.feature2index):
      offset = self.one_hot_encoder.feature_indices_[feature]
    else:
      return False
    idx += offset
    return idx

  def get_property_by_one_hot_encoded_index(self, prop_idx, is_label=False):
    if self.one_hot_encoder is None:
      return False
    if not is_label:
      i = bisect.bisect_right(self.one_hot_encoder.feature_indices_, prop_idx)-1
      idx = prop_idx - self.one_hot_encoder.feature_indices_[i]
    else:
      i = -1
      idx = prop_idx - 0
    return self.index2feature[i].get(idx)


if __name__ == '__main__':
  src = '../Data/watermelon/watermelon_3.0.csv'
  feature_types = [0,0,0,0,0,0,1,1,0]
  feature_maker = FeatureMaker(src=src, delimiter=',', types=feature_types, norm=True)
  x, y = feature_maker.make(skip_rows=1, skip_cols=1)
  print(feature_maker.shuffle(x, y))
  print(json.dumps(feature_maker.index2feature, indent=2, ensure_ascii=False))

  src = '../Data/watermelon/watermelon_2.0.csv'
  feature_types = [0,0,0,0,0,0,0]
  feature_maker = FeatureMaker(src=src, delimiter=',', types=feature_types, norm=True)
  x, y = feature_maker.make(skip_rows=1, skip_cols=1)
  print(feature_maker.train_test_split(x, y, test_size=5))
  print(json.dumps(feature_maker.index2feature, indent=2, ensure_ascii=False))

