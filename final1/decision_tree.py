import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# A decision tree node
class Node:
    def __init__(self, gini, num_samples, num_samples_per_class, pred_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = pred_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None
        self.root = None


class DTree:
    def __init__(self, max_depth, min_size):
        self.max_depth = max_depth
        self.min_size = min_size
        self.root = None

    # Calculate the Gini index for a split dataset
    def __gini_index(self, groups, classes):
        # count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))

        # Sum weighted Gini index for each group
        gini = 0.0
        for group in groups:
            size = float(len(group))

            # Avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # Score the group based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += (p * p)
            # Weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances)
        return gini

    # Split a dataset based on an attribute and an attribute value
    def __test_split(self, index, value, data):
        left = []
        right = []
        for row in data:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    # Select the best split point for a dataset
    def __get_split(self, data):
        class_values = list(set(row[-1] for row in data))
        print(class_values)
        b_index = 999
        b_value = 999
        b_score = 999
        b_groups = None

        for index in range(len(data[0])-1):
            for row in data:
                groups = self.__test_split(index, row[index], data)
                gini = self.__gini_index(groups, class_values)
                if gini < b_score:
                    b_index = index
                    b_value = row[index]
                    b_score = gini
                    b_groups = groups

        gini = b_score
        num_samples = len(data)
        num_samples_per_class = [len(i) for i in b_groups]
        pred_class = np.argmax(num_samples_per_class)
        node = Node(gini, num_samples, num_samples_per_class, pred_class)
        node.feature_index = b_index
        node.threshold = b_value
        return node

    # Fit the training data, build the decision tree
    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        train = pd.concat([X, y], axis=1)
        train = train.to_numpy()
        self.root = self.__get_split(train)
        # self.__split(self.root, self.max_depth, self.min_size, 1)
        return self.root


data = pd.read_csv('data/iris.csv')
target = 'variety'
labelencoder = LabelEncoder()
data[target] = labelencoder.fit_transform(data[target])
features = data.columns.drop(target)
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)


"""
train = pd.read_csv('data/test.csv')
target = 'Y'
features = train.columns.drop(target)
X = train[features]
y = train[target]
"""


tree = DTree(max_depth=1, min_size=1)
clf = tree.fit(X, y)
print('gini = ', clf.gini)
print('# samples = ', clf.num_samples)
print('# samples/class = ', clf.num_samples_per_class)
print('pred class = ', clf.predicted_class)
print('feature_index = ', clf.feature_index)
print('threshold = ', clf.threshold)

#tree.print_tree(clf)
