import pandas as pd
from decision_tree import DTreeClassifier

train = pd.read_csv('data/test.csv')
target = 'Y'
features = train.columns.drop(target)
X = train[features]
y = train[target]
feature_names = list(X.columns)
class_names = list(set(y))

tree = DTreeClassifier(max_depth=1)
clf = tree.fit(X, y)

test = X.iloc[:3]
print(test)
y_pred = tree.predict(test)
print(y_pred)

tree.print_tree_dot(clf, feature_names, class_names)
