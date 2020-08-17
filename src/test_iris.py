import pandas as pd
from decision_tree import DTree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/iris.csv')
target = 'variety'
labelencoder = LabelEncoder()
data[target] = labelencoder.fit_transform(data[target])
features = data.columns.drop(target)
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

tree = DTree(max_depth=5, min_size=2)
clf = tree.fit(X_train, y_train)
tree.print_tree(clf)

y_pred = tree.predict(X_test)
accuracy_score = tree.accuracy(y_test, y_pred)
print('Accuracy Score: ', accuracy_score)
