import math
import itertools

# DO NOT CHANGE THE FOLLOWING LINE
def apriori(itemsets, threshold):
    # Helper function that calculates the support of an itemset
    def support_count(itemset, transactions):
        """Calculate the support for an itemset as the proportion of transactions that contain this itemset."""
        return sum(1 for trans in transactions if itemset <= trans) / len(transactions)

    # Initial candidates from single items in transactions
    candidates = {frozenset([item]) for trans in itemsets for item in trans}
    support = {itemset: support_count(itemset, itemsets) for itemset in candidates}
    # Filtering candidates by support threshold
    frequent = {itemset for itemset, sup in support.items() if sup >= threshold / 100}

    all_frequent = set(frequent)  # To store all levels of frequent itemsets
    k = 2  # Start checking for itemsets of size 2
    while True:
        # Generate new candidates from combinations of the frequent itemsets
        new_candidates = set(frozenset(i.union(j)) for i in frequent for j in frequent if len(i.union(j)) == k)
        if not new_candidates:
            break
        support_new = {itemset: support_count(itemset, itemsets) for itemset in new_candidates}
        support.update(support_new)
        current_frequent = {itemset for itemset, sup in support_new.items() if sup >= threshold / 100}
        if not current_frequent:
            break
        frequent = current_frequent
        all_frequent.update(frequent)
        k += 1

    # Format the result as a list of pairs (itemset, support_percentage)
    return [(set(itemset), round(sup * 100, 2)) for itemset, sup in support.items() if itemset in all_frequent]

# DO NOT CHANGE THE FOLLOWING LINE
def association_rules(itemsets, frequent_itemsets, metric, metric_threshold):
    # DO NOT CHANGE THE PRECEDING LINE
    
    # Should return a list of triples: condition, effect, metric value 
    # Each entry (c,e,m) represents a rule c => e, with the matric value m
    # Rules should only be included if m is greater than the given threshold.    
    # e.g. [(set(condition),set(effect),0.45), ...]
    return []
