from sklearn.datasets import load_iris
from sklearn import tree
from graphviz import Source

clf = tree.DecisionTreeClassifier()
iris = load_iris()

clf = clf.fit(iris.data, iris.target)
dot_str = tree.export_graphviz(clf, out_file=None)
print(dot_str)
src = Source(dot_str)
# print(help(src.render))
src.render(view=True, cleanup=True)

