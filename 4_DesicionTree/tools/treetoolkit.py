# -*-coding:utf-8-*-
import sys
import matplotlib.pyplot as plt
import matplotlib.font_manager as fontmanager
import traceback
import os
ABS_PATH=os.path.dirname(os.path.realpath(__file__))
print(ABS_PATH)

class  DNode:
  def __init__(self, samples, gain=0.0, fea_name="default", fea_type="default", fea_value=0.0):
    self.fea_name = fea_name
    self.fea_value = fea_value
    self.fea_type = fea_type
    self.gain = gain
    self.samples = samples
    self.children = None
    self.is_leaf = False
  def set_leaf(self, label):
    self.is_leaf = True
    self.fea_name = "leaf"
    self.fea_value = label
    self.fea_type = "discrete"
  def next_node(self, value):
    if self.children==None:
      return None
    if self.fea_type=="discrete":
      return self.children.get(str(value))
    else:
      key = "<="+str(self.fea_value) if value<=self.fea_value else ">"+str(self.fea_value)
      return self.children.get(key) 
  def __str__(self):
    samples = "samples:%d"%self.samples
    if self.is_leaf:
      value = "label:"+str(self.fea_value)
      desc = '\n'.join([samples, value])
    else:
      name = "name:"+str(self.fea_name)
      gain = "gain:"+("%.4f"%self.gain)
      value = "" if self.fea_type=="discrete" else "<="+str(self.fea_value)
      desc = '\n'.join([name+value, gain, samples])
    return desc
  def append_child(self, child, value):
    if self.children==None:
      self.children = {}
    if self.fea_type=="discrete":
      self.children[str(value)] = child
    else:
      if value<=self.fea_value:
        self.children["<="+str(self.fea_value)] = child
      else:
        self.children[">"+str(self.fea_value)] = child
  def traverse_pre_order_recursively(self, do):
    do(self)
    if self.children==None:
      return
    for it in self.children:
      self.children[it].traverse_pre_order_recursively(do=do)

def plot_dtree(root, level_interval=1, sibling_interval=3):
  if root==None:
    return
  microsoft_yahei = fontmanager.FontProperties(fname=ABS_PATH+"/microsoft-yahei.ttf")
  ## arrow style
  arrow_props = dict(arrowstyle="<-")
  ## plot root
  root_xy = (2.0,6.0)
  an_root = plt.annotate(s=str(root).decode('utf-8'), xy=root_xy, xytext=root_xy, fontproperties=microsoft_yahei)
  ## canvas size
  center_x = root_xy[0]
  left, top = root_xy
  right, bottom = root_xy
  ## BFS plot
  q = [(root,an_root)]
  while len(q)>0:
    next_q = []
    for node, parent_an in q:
      ## put (child, parent-xytext)
      if node.children!=None:
        next_q.extend([(n, node.children.get(n), parent_an) for n in node.children])
    if len(next_q)==0:
      break
    ## compute center of next_q
    half_w = (len(next_q)-1)*(sibling_interval)/2
    x = center_x-half_w
    left = x if x<left else left
    bottom -= level_interval
    y = bottom
    q = []
    for arrow_text, child, parent_an in next_q:
      parent_xy = parent_an.get_position()
      an_child = plt.annotate(s=str(child).decode('utf-8'), xy=parent_xy, xytext=(x,y), arrowprops=arrow_props, fontproperties=microsoft_yahei)
      q.append((child,an_child))
      ## plot condition on arrow between parent and child
      mid_x = (parent_xy[0]-x)/2.0+x
      mid_y = (parent_xy[1]-y)/2.0+y
      plt.text(mid_x, mid_y, arrow_text.decode('utf-8'), fontproperties=microsoft_yahei)
      x += sibling_interval
    right = x if x>right else right
  plt.xlim(left-sibling_interval/5, right+sibling_interval/5)
  plt.ylim(bottom-level_interval, top+level_interval)
  plt.axis('off')
  plt.show()


if __name__=="__main__":
  samples = 2
  dnode = DNode(samples)
  dnode.set_leaf(label=0)
  print(dnode)
  dnode = DNode(samples, 1.223, "test", "continuos", 2.3)
  print(dnode)
