# Implementation of the CART algorithm to train decision tree classifiers.
import numpy as np
import sys


# A decision tree node
class Node:
    class_counter = 0

    def __init__(self, gini, num_samples, num_samples_per_class, pred_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.pred_class = pred_class
        self.feature_index = 0
        self.threshold = 0
        self.index = Node.class_counter
        self.left = None
        self.right = None
        Node.class_counter += 1


class DTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    # Fit the model using data in a dataframe
    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        X = X.to_numpy()
        self.tree_ = self._build_tree(X, y)
        return self.tree_

    # Make a prediction
    def predict(self, X):
        X = X.to_numpy()
        return [self._predict(inputs) for inputs in X]

    # Calculate the accuracy
    def accuracy(self, actual, predicted):
        act = actual.to_numpy()
        correct = 0
        for i in range(len(act)):
            if act[i] == predicted[i]:
                correct += 1
        return correct / float(len(act)) * 100

    # Print a decision tree
    def print_tree(self, node, feature_names, depth=0):
        self.feature_names = feature_names
        indent = ' ' * (depth * 5)
        if node is not None:
            is_leaf = not node.right
            if is_leaf:
                print(indent, node.pred_class)
            else:
                print(indent, feature_names[node.feature_index], '<', node.threshold)
            print(indent, 'gini=', node.gini)
            print(indent, 'samples=', node.num_samples)
            print(indent, 'samples/class', node.num_samples_per_class)
            print(' ')

            self.print_tree(node.left, feature_names, depth + 1)
            self.print_tree(node.right, feature_names, depth + 1)

    # Print a dot decision tree
    def print_tree_dot(self, node, feature_names, class_names):
        dot_file = open('data/output.dot', 'w')
        self._print_tree(node, feature_names, class_names, dot_file)
        print('}', file=dot_file)
        dot_file.close()

    # Traverse the tree breadth-first, printing dot code for each node
    def _print_tree(self, node, feature_names, class_names, dot_file, depth=0):
        output_str = ''
        if depth == 0:
            print('digraph Tree {', file=dot_file)
            print('node [shape=box] ;', file=dot_file)
        self.feature_names = feature_names
        if node is not None:
            is_leaf = not node.right
            if is_leaf:
                output_str = str(node.index) + ' ' + \
                             '[label=\"' + \
                             str(class_names[node.pred_class]) + '\\'
            else:
                output_str = str(node.index) + ' ' + \
                             '[label=\"' + \
                             feature_names[node.feature_index] + ' < ' + str(node.threshold) + '\\'
            output_str += 'ngini = ' + str(node.gini) + '\\' + \
                          'nsamples = ' + str(node.num_samples) + '\\' + \
                          'nvalue = ' + str(node.num_samples_per_class) + \
                          '\"] ;'
            print(output_str, file=dot_file)
            if is_leaf is False:
                print(str(node.index) + ' -> ' + str(node.left.index), file=dot_file)
            self._print_tree(node.left, feature_names, class_names, dot_file, depth + 1)
            if is_leaf is False:
                print(str(node.index) + ' -> ' + str(node.right.index), file=dot_file)
            self._print_tree(node.right, feature_names, class_names, dot_file, depth + 1)

    # Compute the gini
    def _gini(self, y):
        size = y.size
        return 1.0 - sum((np.sum(y == c) / size) ** 2 for c in range(self.n_classes_))

    # Find the best split
    def _best_split(self, X, y):
        size = y.size
        if size <= 1:
            return None, None

        # Count of each class in the current node
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]

        # Gini of current node.
        best_gini = 1.0 - sum((n / size) ** 2 for n in num_parent)
        best_idx, best_thr = None, None

        # Loop through all features.
        for idx in range(self.n_features_):
            # Sort data along selected feature.
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))

            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, size):
                c = int(classes[i - 1])
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (size - i)) ** 2 for x in range(self.n_classes_)
                )

                # The gini of a split is the weighted average of the gini
                # impurity of the children.
                gini = (i * gini_left + (size - i) * gini_right) / size

                # Don't split identical values
                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        return best_idx, best_thr

    # Build the decision tree recursively finding the best split
    def _build_tree(self, X, y, depth=0):
        # Population for each class in current node.
        # The predicted class is the one with the largest population
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(gini=self._gini(y),
                    num_samples=y.size,
                    num_samples_per_class=num_samples_per_class,
                    pred_class=predicted_class
                    )

        # Split recursively until maximum depth is reached.
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._build_tree(X_left, y_left, depth + 1)
                node.right = self._build_tree(X_right, y_right, depth + 1)
        return node

    # Predict class for a single sample
    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.pred_class
