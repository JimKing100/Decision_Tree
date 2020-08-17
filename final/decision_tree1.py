# Implementation of the CART algorithm to train decision tree classifiers.
import numpy as np


# A decision tree node
class Node:
    def __init__(self, gini, num_samples, num_samples_per_class, pred_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.pred_class = pred_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

    # Print a decision tree
    def print_tree(self, feature_names, class_names, show_details):
        lines, _, _, _ = self._print_aux(
            feature_names, class_names, show_details, root=True
        )
        for line in lines:
            print(line)

    # See https://stackoverflow.com/a/54074933/1143396 for similar code.
    def _print_aux(self, feature_names, class_names, show_details, root=False):
        is_leaf = not self.right
        if is_leaf:
            lines = [class_names[self.pred_class]]
        else:
            lines = [
                "{} < {:.2f}".format(feature_names[self.feature_index], self.threshold)
            ]
        if show_details:
            lines += [
                "gini = {:.2f}".format(self.gini),
                "samples = {}".format(self.num_samples),
                str(self.num_samples_per_class),
            ]
        width = max(len(line) for line in lines)
        height = len(lines)
        if is_leaf:
            lines = ["║ {:^{width}} ║".format(line, width=width) for line in lines]
            lines.insert(0, "╔" + "═" * (width + 2) + "╗")
            lines.append("╚" + "═" * (width + 2) + "╝")
        else:
            lines = ["│ {:^{width}} │".format(line, width=width) for line in lines]
            lines.insert(0, "┌" + "─" * (width + 2) + "┐")
            lines.append("└" + "─" * (width + 2) + "┘")
            lines[-2] = "┤" + lines[-2][1:-1] + "├"
        width += 4  # for padding

        if is_leaf:
            middle = width // 2
            lines[0] = lines[0][:middle] + "╧" + lines[0][middle + 1:]
            return lines, width, height, middle

        # If not a leaf, must have two children.
        left, n, p, x = self.left._print_aux(feature_names, class_names, show_details)
        right, m, q, y = self.right._print_aux(feature_names, class_names, show_details)
        top_lines = [n * " " + line + m * " " for line in lines[:-2]]
        # fmt: off
        middle_line = x * " " + "┌" + (n - x - 1) * "─" + lines[-2] + y * "─" + "┐" + (m - y - 1) * " "
        bottom_line = x * " " + "│" + (n - x - 1) * " " + lines[-1] + y * " " + "│" + (m - y - 1) * " "
        # fmt: on
        if p < q:
            left += [n * " "] * (q - p)
        elif q < p:
            right += [m * " "] * (p - q)
        zipped_lines = zip(left, right)
        lines = (
            top_lines
            + [middle_line, bottom_line]
            + [a + width * " " + b for a, b in zipped_lines]
        )
        middle = n + width // 2
        if not root:
            lines[0] = lines[0][:middle] + "┴" + lines[0][middle + 1:]
        return lines, n + m + width, max(p, q) + 2 + len(top_lines), middle


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

    # Print the decision tree
    def print_tree(self, feature_names, class_names, show_details=True):
        self.tree_.print_tree(feature_names, class_names, show_details)

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
