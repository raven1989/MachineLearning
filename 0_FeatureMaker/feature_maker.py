import sys
import numpy as np
import hashlib
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

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
    self.one_hot_encoder = OneHotEncoder(categorical_features=discrete_feature_idx)
    if self.norm:
      self.min_max_scaler = MinMaxScaler()
  def make(self, skip_rows=0, skip_cols=0):
    raw_data = []
    feature_map = [{} for t in self.types]
    with open(self.src) as src:
      for row,line in enumerate(src):
        # print(row, line)
        if row<skip_rows:
          continue
        splitted = line.strip().split(self.delimiter)[skip_cols:]
        raw_sample = []
        for i,t in enumerate(self.types):
          if t==0:
            value = feature_map[i].get(splitted[i])
            if value is None:
              value = float(len(feature_map[i]))
              feature_map[i][splitted[i]] = value
          else:
            value = float(splitted[i])
          raw_sample.append(value)
        raw_data.append(raw_sample)
    raw_data = np.array(raw_data)
    x = raw_data[:,:-1]
    y = raw_data[:,-1:]
    # print(raw_data)
    x = self.one_hot_encoder.fit_transform(X=x, y=y).toarray()
    if self.norm:
      x = self.min_max_scaler.fit_transform(X=x, y=y)
    return x, y

if __name__ == '__main__':
  src = '../Data/watermelon/watermelon_3.0.csv'
  feature_types = [0,0,0,0,0,0,1,1,0]
  feature_maker = FeatureMaker(src=src, delimiter=',', types=feature_types, norm=True)
  print(feature_maker.make(skip_rows=1, skip_cols=1))

  src = '../Data/watermelon/watermelon_2.0.csv'
  feature_types = [0,0,0,0,0,0,0]
  feature_maker = FeatureMaker(src=src, delimiter=',', types=feature_types, norm=True)
  print(feature_maker.make(skip_rows=1, skip_cols=1))

