from collections import Counter
import math

class DecisionTree:
    def __init__(self, max_depth=None):
        """
        Initialize the decision tree.
        
        Args:
            max_depth (int, optional): Maximum depth for tree pruning. Default is None (no pruning).
        """
        self.tree = None
        self.max_depth = max_depth
        self.majority_class = None

    def _entropy(self, y):
        """Calculate entropy of label array y."""
        if not y:
            return 0
        counter = Counter(y)
        entropy = 0.0
        for count in counter.values():
            p = count / len(y)
            entropy -= p * math.log2(p)
        return entropy

    def _information_gain(self, x_column, y):
        """Calculate information gain for a specific feature column."""
        base_entropy = self._entropy(y)
        
        # Calculate weighted entropy of children
        values = Counter(x_column)
        weighted_entropy = 0.0
        
        for value in values:
            subset_indices = [i for i, x in enumerate(x_column) if x == value]
            subset_y = [y[i] for i in subset_indices]
            weighted_entropy += (len(subset_indices) / len(y)) * self._entropy(subset_y)
            
        return base_entropy - weighted_entropy

    def _split_data(self, x, y, split_column):
        """Split the data based on the values in the split column."""
        split_dict = {}
        values = set()
        # Get all unique values for the split column
        for instance in x:
            values.add(instance[split_column])
            
        for value in values:
            indices = [i for i, row in enumerate(x) if row[split_column] == value]
            split_dict[value] = {
                'x': [x[i] for i in indices],
                'y': [y[i] for i in indices]
            }
        return split_dict

    def _build_tree(self, x, y, depth=0):
        """Recursively build the decision tree."""
        # Base cases
        if not x or not y:
            return {"type": "class", "class": self.majority_class}
            
        if len(set(y)) == 1:
            return {"type": "class", "class": y[0]}
        
        if self.max_depth is not None and depth >= self.max_depth:
            majority_class = max(Counter(y).items(), key=lambda x: x[1])[0]
            return {"type": "class", "class": majority_class}

        # Find best split
        n_features = len(x[0])
        best_gain = -1
        best_split = 0
        
        for i in range(n_features):
            feature_values = [row[i] for row in x]
            gain = self._information_gain(feature_values, y)
            if gain > best_gain:
                best_gain = gain
                best_split = i

        # If no information gain, make a leaf node
        if best_gain <= 0:
            majority_class = max(Counter(y).items(), key=lambda x: x[1])[0]
            return {"type": "class", "class": majority_class}

        # Create split node
        split_data = self._split_data(x, y, best_split)
        
        node = {
            "type": "split",
            "split": best_split,
            "children": {}
        }

        # Recursively build child nodes
        for value, subset in split_data.items():
            if subset['x']:
                node["children"][value] = self._build_tree(subset['x'], subset['y'], depth + 1)
            else:
                node["children"][value] = {"type": "class", "class": self.majority_class}

        return node

    def fit(self, x, y):
        """
        Build a decision tree from training data.
        
        Args:
            x: List of training instances
            y: List of training labels
        """
        if not x or not y:
            raise ValueError("Empty training data")
            
        # Store majority class for handling unseen values
        self.majority_class = max(Counter(y).items(), key=lambda x: x[1])[0]
        self.tree = self._build_tree(x, y)

    def _predict_one(self, x):
        """Predict class for a single instance."""
        if not self.tree:
            raise ValueError("Model not fitted. Call fit before predict.")
            
        node = self.tree
        while node["type"] == "split":
            value = x[node["split"]]
            if value not in node["children"]:
                return self.majority_class
            node = node["children"][value]
        return node["class"]

    def predict(self, x):
        """
        Predict classes for multiple instances.
        
        Args:
            x: List of instances to classify
        Returns:
            List of predicted classes
        """
        return [self._predict_one(instance) for instance in x]

    def to_dict(self):
        """Return the tree structure as a dictionary."""
        if not self.tree:
            return {"type": "class", "class": self.majority_class}
        return self.tree