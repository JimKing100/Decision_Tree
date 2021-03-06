import pandas as pd


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

    # Create a terminal node value
    def __to_terminal(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    # Select the best split point for a dataset
    def __get_split(self, data):
        class_values = list(set(row[-1] for row in data))
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
        return{'index': b_index, 'value': b_value, 'groups': b_groups}

    # Create child splits for a node or make terminal
    def __split(self, node, max_depth, min_size, depth):
        left, right = node['groups']
        del(node['groups'])
        # Check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.__to_terminal(left + right)
            return
        # Check for max depth
        if depth >= max_depth:
            node['left'], node['right'] = self.__to_terminal(left), self.__to_terminal(right)
            return
        # Process left child
        if len(left) <= min_size:
            node['left'] = self.__to_terminal(left)
        else:
            node['left'] = self.__get_split(left)
            self.__split(node['left'], self.max_depth, self.min_size, depth + 1)
        # Process right child
        if len(right) <= min_size:
            node['right'] = self.__to_terminal(right)
        else:
            node['right'] = self.__get_split(right)
            self.__split(node['right'], self.max_depth, self.min_size, depth + 1)

    # Print the decision tree
    def print_tree(self, node, depth=0):
        print(node)
        if isinstance(node, dict):
            print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
            self.print_tree(node['left'], depth + 1)
            self.print_tree(node['right'], depth + 1)
        else:
            print('%s[%s]' % ((depth*' ', node)))

    # Fit the training data, build the decision tree
    def fit(self, X, y):
        train = pd.concat([X, y], axis=1)
        train = train.to_numpy()
        self.root = self.__get_split(train)
        self.__split(self.root, self.max_depth, self.min_size, 1)
        return self.root

    # Make a prediction for a single item
    def predict_item(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict_item(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict_item(node['right'], row)
            else:
                return node['right']

    # Make predictions
    def predict(self, X):
        test = X.to_numpy()
        predictions = list()
        for row in test:
            prediction = self.predict_item(self.root, row)
            predictions.append(prediction)
        return predictions

    # Calculate accuracy
    def accuracy(self, actual, predicted):
        act = actual.to_numpy()
        correct = 0
        for i in range(len(act)):
            if act[i] == predicted[i]:
                correct += 1
        return correct / float(len(act)) * 100
