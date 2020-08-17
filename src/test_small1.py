import pandas as pd
from decision_tree1 import DTC


train = pd.read_csv('data/test.csv')
target = 'Y'
features = train.columns.drop(target)
X = train[features]
y = train[target]
feature_names = list(X.columns)
target_names = ['0', '1']

tree = DTC(max_depth=1)
tree.fit(X, y)
# tree.print_tree(clf)

test = X.iloc[:3]
print(test)
y_pred = tree.predict(test)
print(y_pred)

tree.print_tree(list(X.columns), [0, 1])
