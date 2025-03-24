# cardiac_rf.py
import numpy as np
import pandas as pd
from collections import Counter

# Gini Impurity
def gini(y):
    counts = Counter(y)
    impurity = 1.0
    for label in counts:
        prob = counts[label] / len(y)
        impurity -= prob ** 2
    return impurity

# Splitting function
def split_data(X, y, feature_index, threshold):
    left_idx = X[:, feature_index] <= threshold
    right_idx = X[:, feature_index] > threshold

    return X[left_idx], X[right_idx], y[left_idx], y[right_idx]

# Best split
def best_split(X, y):
    best_gain = 0
    best_feature = None
    best_threshold = None
    current_gini = gini(y)

    for feature_index in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_index])

        for threshold in thresholds:
            X_left, X_right, y_left, y_right = split_data(X, y, feature_index, threshold)

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            p = len(y_left) / len(y)
            gain = current_gini - (p * gini(y_left) + (1 - p) * gini(y_right))

            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold

# Decision Tree Node
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# Build Decision Tree
def build_tree(X, y, depth=0, max_depth=10):
    if len(set(y)) == 1 or depth >= max_depth:
        return Node(value=Counter(y).most_common(1)[0][0])

    feature_index, threshold = best_split(X, y)

    if feature_index is None:
        return Node(value=Counter(y).most_common(1)[0][0])

    X_left, X_right, y_left, y_right = split_data(X, y, feature_index, threshold)

    left_child = build_tree(X_left, y_left, depth + 1, max_depth)
    right_child = build_tree(X_right, y_right, depth + 1, max_depth)

    return Node(feature_index, threshold, left_child, right_child)

# Make predictions
def predict_tree(node, x):
    if node.value is not None:
        return node.value

    if x[node.feature_index] <= node.threshold:
        return predict_tree(node.left, x)
    else:
        return predict_tree(node.right, x)

# Random Forest Classifier
class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, sample_size=0.8):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.trees = []

    def fit(self, X, y):
        n_samples = int(len(X) * self.sample_size)

        for _ in range(self.n_trees):
            idxs = np.random.choice(len(X), n_samples, replace=True)
            X_sample, y_sample = X[idxs], y[idxs]
            tree = build_tree(X_sample, y_sample, max_depth=self.max_depth)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([self._predict_single_tree(tree, X) for tree in self.trees])
        predictions = np.swapaxes(predictions, 0, 1)
        final_predictions = [Counter(row).most_common(1)[0][0] for row in predictions]
        return np.array(final_predictions)

    def _predict_single_tree(self, tree, X):
        return np.array([predict_tree(tree, x) for x in X])
