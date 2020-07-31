# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))

    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += (p * p)
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini


# test Gini values
# print(gini_index([[[1, 1], [1, 0]], [[1, 1], [1, 0]]], [0, 1]))
# print(gini_index([[[1, 0], [1, 0]], [[1, 1], [1, 1]]], [0, 1]))


# Split a dataset based on an attribute and an attribute value
def test_split(index, value, data):
    left = []
    right = []
    for row in data:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Select the best split point for a dataset
def get_split(data):
    class_values = list(set(row[-1] for row in data))
    b_index = 999
    b_value = 999
    b_score = 999
    b_groups = None

    for index in range(len(data[0])-1):
        for row in data:
            groups = test_split(index, row[index], data)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index = index
                b_value = row[index]
                b_score = gini
                b_groups = groups

    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# Create a terminal node value
def to_terminal(group):
    print('group = ', group)
    # outcomes = list(set(row[-1] for row in group))
    outcomes = [row[-1] for row in group]
    print('outcomes', outcomes)
    # key = outcomes.count
    # print('key = ', key)
    return max(set(outcomes), key=outcomes.count)


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)


# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root


# Print a decision tree
def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
        print_tree(node['left'], depth+1)
        print_tree(node['right'], depth+1)
    else:
        print('%s[%s]' % ((depth*' ', node)))


data = [[2.771244718,1.784783929,0],
        [1.728571309,1.169761413,0],
        [3.678319846,2.81281357,0],
        [3.961043357,2.61995032,0],
        [2.999208922,2.209014212,0],
        [7.497545867,3.162953546,1],
        [9.00220326,3.339047188,1],
        [7.444542326,0.476683375,1],
        [10.12493903,3.234550982,1],
        [6.642287351,3.319983761,1]]
# split = get_split(data)
# print('Split: [X%d < %.3f]' % ((split['index']+1), split['value']))
tree = build_tree(data, 3, 1)
print_tree(tree)
