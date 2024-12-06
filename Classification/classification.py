import numpy
import math
from collections import Counter

# Determine if all values in a list are the same 
# Useful for the second basecase above
def same(values):
    """Determine if all values in a list are the same"""
    if not values: return True
    # if there are values:
    # pick the first, check if all other are the same 
    first = values[0]
    return all(v == first for v in values)

# Determine how often each value shows up 
# in a list; this is useful for the entropy
# but also to determine which values is the 
# most common
def counts(values):
    """Count occurrences of each value in the list"""
    return Counter(values)
   
# Return the most common value from a list 
# Useful for base cases 1 and 3 above.
def majority(values):
    """Return the most common value from a list"""
    if not values:
        return None
    # count the values
    count_dict = counts(values)
    # return the most common value
    return max(count_dict.items(), key=lambda x: x[1])[0]

# Calculate the entropy of a set of values 
# First count how often each value shows up 
# When you divide this value by the total number 
# of elements, you get the probability for that element 
# The entropy is the negation of the sum of p*log2(p) 
# for all these probabilities.
def entropy(values):
    """Calculate entropy of a list of values"""
    if not values:
        return 0
    # count the values 
    counts_dict = counts(values)
    # calculate the length of the list
    total = len(values)
    # calculate the probabilities
    probabilities = [count/total for count in counts_dict.values()]
    # return the entropy
    return -sum(p * math.log2(p) for p in probabilities)

def calculate_information_gain(column_values, ys, current_entropy):
    """Calculate information gain for a potential split"""
    # Group by values
    groups = {}
    for i, value in enumerate(column_values):
        if value not in groups:
            groups[value] = []
        groups[value].append(ys[i])
        
    # Calculate weighted entropy after split
    weighted_entropy = 0
    total = len(ys)
    
    for group in groups.values():
        weight = len(group) / total
        weighted_entropy += weight * entropy(group)
        
    # calculate the information gain by subtracting the weighted entropy from the current entropy
    return current_entropy - weighted_entropy

# Create the decision tree recursively using Hunt's Algorithm
def make_node(previous_ys, xs, ys, columns):
    """
    Implementation of Hunt's Algorithm for decision tree construction.
    
    Args:
        previous_ys: Class labels from parent node (for majority class when no samples).
        xs: Feature values.
        ys: Class labels.
        columns: Available columns for splitting.

    Returns:
        A dictionary representing the decision tree node.
    """
    # print(f"Recursive call - Dataset rows (xs): {xs}")
    # print(f"Dataset dimensions: {len(xs)} rows, {len(xs[0]) if xs else 0} columns")
    # print(f"Columns available: {columns}")
    
    # Hunt's Algorithm Base Casess:
    # Case 1: No samples left
    if not xs or not ys:
        return {"type": "class", "class": majority(previous_ys)}
    # Case 2: All samples belong to same class  
    if same(ys):
        return {"type": "class", "class": ys[0]}
    # Case 3: No attributes left but still samples from different classes
    columns = [col for col in columns if col < len(xs[0])]
    if not columns:
        return {"type": "class", "class": majority(ys)}
    
    # Hunt's Algorithm Recursive Case:
    # 1. Select best attribute to split on
    # Calculate current entropy
    current_entropy = entropy(ys)
    best_gain = -1
    best_column = None
    
    # Find the best column to split
    for col in columns:
        # Get values for this column
        column_values = [row[col] for row in xs]
        # Calculate information gain
        gain = calculate_information_gain(column_values, ys, current_entropy)
        if gain > best_gain:
            best_gain = gain
            best_column = col
    
    # If no gain, return a leaf node with the majority class
    if best_gain <= 0:
        return {"type": "class", "class": majority(ys)}
    
    # 2. Create node that splits on best attribute
    node = {"type": "split", "split": best_column, "children": {}}
    value_groups = {}
    
    # 3. Partition data based on best attribute values (best column)
    for i, row in enumerate(xs):
        value = row[best_column]
        if value not in value_groups:
            value_groups[value] = {"xs": [], "ys": []}
        new_row = row[:best_column] + row[best_column + 1:]
        value_groups[value]["xs"].append(new_row)
        value_groups[value]["ys"].append(ys[i])
    
    # 4. Recursively apply Hunt's Algorithm to each partition
    new_columns = [c for c in columns if c != best_column]
    if not new_columns:
        return {"type": "class", "class": majority(ys)}
    # Create child nodes
    for value, group in value_groups.items():
        if not group["xs"]:  # Handle empty partitions
            node["children"][value] = {"type": "class", "class": majority(ys)}
        else:
            node["children"][value] = make_node(ys, group["xs"], group["ys"], new_columns)
    
    return node

# This is the main decision tree class 
# DO NOT CHANGE THE FOLLOWING LINE
class DecisionTree:
# DO NOT CHANGE THE PRECEDING LINE
    #Constructor to initialize the decision tree.
    def __init__(self, tree={}):
        self.tree = tree #Holds the decision tree struction.
        self.majority = None #Stores the majority class from the training data.       
    # DO NOT CHANGE THE FOLLOWING LINE    
    def fit(self, x, y):
    # DO NOT CHANGE THE PRECEDING LINE
        """
        Trains the decision tree using the training data (x) and labels (y).
        
        args:
            x: 2D list of features (rows are instances, columns are attributes).
            y: List of labels corresponding to each instance in x.
        """
        # Calculate the majority class from the training labels `y`.      
        self.majority = majority(y)
        # Build the decision tree by creating the root node and recursively splitting the data.       
        self.tree = make_node(y, x, y, list(range(len(x[0]))))
        
    # DO NOT CHANGE THE FOLLOWING LINE    
    def predict(self, x):
    # DO NOT CHANGE THE PRECEDING LINE 
        """
        Predicts the class for a set of instances `x` using the decision tree.

        Args:
            x: List of instances, where each instance is a list of attribute values.

        Returns:
            List: Class predictions for each instance in `x`. Returns `None` if the tree is not built.
        """
        # Check if the tree exists. If the tree is empty (not trained), return None.   
        if not self.tree:
            return None

        def traverse(node, instance):
            """
            Helper function to traverse the decision tree recursively and classify an instance.

            Args:
                node (dict): The current node in the decision tree.
                instance (list): The attributes of the instance being classified.

            Returns:
                The predicted class if a leaf node is reached, or the majority class if no
                appropriate child node exists for an attribute value.
            """
            # Traverse the decision tree based on the instance's attributes.
            # Start with the root as the "current" node.
            # Continue traversing as long as the node is a 'split' node.
            while node['type'] == 'split':
                # Get the attribute index the node splits on.
                attribute_index = node['split'] 
                # Get the value of that attribute for the current instance.
                attribute_value = instance[attribute_index] 
                # If there's no child for the attribute value, return the majority class.
                if attribute_value not in node['children']:
                    return self.majority
                # Otherwise, move the relevant child node based on the attribute value.
                node = node['children'][attribute_value]
            # When a leaf node is reached, return its class label.
            return node['class']
        # Apply the traverse function to each instance in x and return the predictions as a list.
        return [traverse(self.tree, instance) for instance in x]  
    
    # DO NOT CHANGE THE FOLLOWING LINE
    def to_dict(self):
    # DO NOT CHANGE THE PRECEDING LINE
        # change this if you store the tree in a different format
        """
        Converts the decision tree into a dictionary representation.

        Returns:
            dict: A dictionary representation of the decision tree.
        """
        # This method returns the internal tree structure as a dictionary.
        return self.tree
       