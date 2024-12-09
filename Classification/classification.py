from collections import Counter
import math

class DecisionTree:
    
    def __init__(self, max_depth=None):
        self.tree = None
        self.max_depth = max_depth
        self.majority_class = None


    def _entropy(self, y):
        # we calculate the entropy of the label array y
        if not y:
            return 0
        counter = Counter(y)
        entropy = 0.0
        for count in counter.values():
            p = count / len(y)
            entropy -= p * math.log2(p)
        return entropy

    def _information_gain(self, x_column, y):
        # we calculate the info gain on a specific column 
        base_entropy = self._entropy(y)
        
        # Calculate weighted entropy of all the children
        values = Counter(x_column)
        weighted_entropy = 0.0
        
        for value in values:
            subset_indices = [i for i, x in enumerate(x_column) if x == value]
            subset_y = [y[i] for i in subset_indices]
            weighted_entropy += (len(subset_indices) / len(y)) * self._entropy(subset_y)
            
        # and then this is the info gain - the entropy delta basically     
        return base_entropy - weighted_entropy

    def _split_data(self, x, y, split_column):
        # this is a helper method and we split the data, this method is used in build tree
        split_dict = {}
        values = set()
        # Gets all unique values for the split column, and adds them to the values set. the split dict comes as well and makes a dict
        # for splitting, with our x for each x and a the y

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
        # this is of course a recursive process here.
        # Base cases are here - if there is no more x or y then we get the majority class
        if not x or not y:
            return {"type": "class", "class": self.majority_class}
            
        # another base case if we only have one type of y then it is pure so we stop 
        if len(set(y)) == 1:
            return {"type": "class", "class": y[0]}
        
        # and then we would also stop if we have a max depth here. 
        if self.max_depth is not None and depth >= self.max_depth:
            majority_class = max(Counter(y).items(), key=lambda x: x[1])[0]
            return {"type": "class", "class": majority_class}

        # Find best split. the for loop accomplishes that task
        n_features = len(x[0])
        best_gain = -1
        best_split = 0
        
        for i in range(n_features):
            feature_values = [row[i] for row in x]
            gain = self._information_gain(feature_values, y)
            if gain > best_gain:
                best_gain = gain
                best_split = i

        # If no information gain, make a leaf node. Thus this is another base case. We need a certain amount of gain
        if best_gain <= 0:
            majority_class = max(Counter(y).items(), key=lambda x: x[1])[0]
            return {"type": "class", "class": majority_class}

        # Create split node. This is where that helper method comes into play. 
        split_data = self._split_data(x, y, best_split)
        
        # define that node here 
        node = {
            "type": "split",
            "split": best_split,
            "children": {}
        }

        # Recursively builds those child nodes using this for loop, we add to split data items and continue to loop through it and add
        # thus making this the key aspect of the recursive process here. 
        for value, subset in split_data.items():
            if subset['x']:
                # if x has a subset in it then we continue to build the tree from there
                node["children"][value] = self._build_tree(subset['x'], subset['y'], depth + 1)
            else:
                # this is essentially a base case
                node["children"][value] = {"type": "class", "class": self.majority_class}

        return node


    def fit(self, x, y):
        # straight forward here we just use our methods.
        if not x or not y:
            raise ValueError("Empty training data")
            
        # Store majority class, this is used for handling unseen values as seen
        self.majority_class = max(Counter(y).items(), key=lambda x: x[1])[0]
        # and then we build the tree
        self.tree = self._build_tree(x, y)

    # this is to prodict a value for a single instance, it is a helper method to predict. we call this for each instance in x
    def _predict_one(self, x):
        # added this in here to help debug 
        if not self.tree:
            raise ValueError("Model not fitted. Call fit before predict.")
            
        # basically does one split bcuz we only have one feature
        node = self.tree
        while node["type"] == "split":
            value = x[node["split"]]
            if value not in node["children"]:
                return self.majority_class
            node = node["children"][value]
        return node["class"]

    def predict(self, x):
        
        # x: list of instances to classify
        
        # will return a list of predicted classes
        
        return [self._predict_one(instance) for instance in x]

    # simple to dict here. 
    def to_dict(self):
        """Return the tree structure as a dictionary."""
        if not self.tree:
            return {"type": "class", "class": self.majority_class}
        return self.tree