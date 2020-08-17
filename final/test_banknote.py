import pandas as pd
from decision_tree import DTreeClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/data_banknote_authentication.csv')
target = 'class'
data[target] = data[target].astype(float)
features = data.columns.drop(target)
X = data[features]
y = data[target]
feature_names = list(X.columns)
class_names = list(set(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

tree = DTreeClassifier(max_depth=5)
clf = tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)
accuracy_score = tree.accuracy(y_test, y_pred)
print('Accuracy Score: ', accuracy_score)

# tree.print_tree(clf, feature_names)
tree.print_tree_dot(clf, feature_names, class_names)
