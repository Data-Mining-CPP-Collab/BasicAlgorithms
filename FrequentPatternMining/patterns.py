import math
import itertools
from itertools import combinations


# DO NOT CHANGE THE FOLLOWING LINE
def apriori(itemsets, threshold):
    # DO NOT CHANGE THE PRECEDING LINE

    def calculate_support(itemset, transactions):
        # we need to get the support here, fairly straight forward
        count = sum(1 for transaction in transactions if itemset.issubset(transaction))
        return count / len(transactions)

    def generate_candidates(prev_frequent_itemsets, k):
        # then we generate candidates, here is the method with the k value
        items = {item for itemset in prev_frequent_itemsets for item in itemset}
        return [set(comb) for comb in combinations(items, k)]

    # create the transaction set using the itemsets parameter
    transactions = [set(transaction) for transaction in itemsets]
    # here is what we will return
    frequent_itemsets = []

    # Generate 1-itemsets
    candidates = [{item} for transaction in transactions for item in transaction]
    # the next line creates the list of candidates and uses a map of set to antoehr set 
    candidates = list(map(set, set(map(frozenset, candidates))))  
    k = 1

    # here is the main loop 
    while candidates:
        # Calculate support for all the candidates
        supports = [(itemset, calculate_support(itemset, transactions)) for itemset in candidates]
        # and then we filter by support threshold
        current_frequent_itemsets = [(itemset, support) for itemset, support in supports if support >= threshold]
        # Append to the return 
        frequent_itemsets.extend(current_frequent_itemsets)

        # Generate next set
        candidates = generate_candidates([itemset for itemset, _ in current_frequent_itemsets], k + 1)
        k += 1

    # returning 
    return frequent_itemsets

# DO NOT CHANGE THE FOLLOWING LINE
def association_rules(itemsets, frequent_itemsets, metric, metric_threshold):
    rules = []

    # convert frequent_itemsets to a dictionary for quick lookup
    # the dict has the item and its support in key value pairs for faster lookup
    frequent_dict = {frozenset(item): support for item, support in frequent_itemsets}

    for itemset, support_AB in frequent_itemsets:
        itemset = frozenset(itemset)

        # generate all possible not empty proper subsets, which are gonna be the antecedents
        # the subset != itemset ensures that there is an antecedent here, and thus an actual rule is made
        subsets = [frozenset(subset) for subset in powerset(itemset) if subset and subset != itemset]

        for antecedent in subsets:
            consequence = itemset - antecedent
            if not consequence:
                continue

            #get the supports for a and B.
            support_A = frequent_dict[antecedent]
            support_B = frequent_dict[consequence]

            # Calculate metrics. The ones we didnt talk about were read about and implemented
            confidence = support_AB / support_A if support_A > 0 else 0
            lift = confidence / support_B if support_B > 0 else 0
            kulczynski = 0.5 * (confidence + support_AB / support_B) if support_B > 0 else 0
            cosine = support_AB / (support_A * support_B)**0.5 if support_A > 0 and support_B > 0 else 0

            # get the selected metric value with this
            metrics = {
                "confidence": confidence,
                "lift": lift,
                "kulczynski": kulczynski,
                "cosine": cosine,
            }

            if metric == "max":
                # Use the maximum metric value
                max_metric = max(metrics.values())
                if max_metric > metric_threshold:
                    rules.append((antecedent, consequence, max_metric))
            elif metric == "all":
                # Include rules for all metrics above the threshold
                for metric_name, value in metrics.items():
                    if value > metric_threshold:
                        rules.append((antecedent, consequence, value))
            else:
                # Specific metric
                metric_value = metrics.get(metric, None)
                if metric_value is not None and metric_value > metric_threshold:
                    rules.append((antecedent, consequence, metric_value))

    return rules

def powerset(iterable):
    """
    Returns all possible subsets of an iterable as a list of tuples
    Using this code with the library
    """
    from itertools import chain, combinations
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
