from sklearn.datasets import load_iris
from sklearn import tree
import graphviz


clf = tree.DecisionTreeClassifier()
iris = load_iris()
clf = clf.fit(iris.data, iris.target)
tree.export_graphviz(clf, out_file='output.dot')

with open("output.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)
