import pandas as pd
from decision_tree import DTree

train = pd.read_csv('data/test.csv')
target = 'Y'
features = train.columns.drop(target)
X = train[features]
y = train[target]

tree = DTree(max_depth=1, min_size=1)
clf = tree.fit(X, y)
tree.print_tree(clf)

test = X.iloc[:3]
print(test)
y_pred = tree.predict(test)
print(y_pred)
