def association_rules(itemsets, frequent_itemsets, metric, metric_threshold):
    rules = []
    
    # Helper function to calculate support
    def calc_support(itemset):
        return sum(1 for trans in itemsets if itemset <= trans) / len(itemsets)
    
    # Helper function to calculate confidence
    def calc_confidence(antecedent, consequent):
        ant_sup = calc_support(antecedent)
        if ant_sup == 0:
            return 0
        return calc_support(antecedent | consequent) / ant_sup
    
    # Helper function to calculate lift
    def calc_lift(antecedent, consequent):
        conf = calc_confidence(antecedent, consequent)
        cons_sup = calc_support(consequent)
        if cons_sup == 0:
            return 0
        return conf / cons_sup
    
    # Helper function to calculate kulczynski
    def calc_kulczynski(antecedent, consequent):
        conf_a_c = calc_confidence(antecedent, consequent)
        conf_c_a = calc_confidence(consequent, antecedent)
        return (conf_a_c + conf_c_a) / 2
    
    # Helper function to calculate cosine
    def calc_cosine(antecedent, consequent):
        both_sup = calc_support(antecedent | consequent)
        ant_sup = calc_support(antecedent)
        cons_sup = calc_support(consequent)
        if ant_sup * cons_sup == 0:
            return 0
        return both_sup / math.sqrt(ant_sup * cons_sup)
    
    # Helper function to calculate max confidence
    def calc_max_confidence(antecedent, consequent):
        return max(calc_confidence(antecedent, consequent),
                  calc_confidence(consequent, antecedent))
    
    # For each frequent itemset
    for itemset, support in frequent_itemsets:
        # Generate all possible non-empty proper subsets
        itemset = frozenset(itemset)
        for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                
                # Calculate the appropriate metric
                metric_value = 0
                if metric == "lift":
                    metric_value = calc_lift(antecedent, consequent)
                elif metric == "all":
                    metric_value = min(calc_confidence(antecedent, consequent),
                                    calc_confidence(consequent, antecedent))
                elif metric == "max":
                    metric_value = calc_max_confidence(antecedent, consequent)
                elif metric == "kulczynski":
                    metric_value = calc_kulczynski(antecedent, consequent)
                elif metric == "cosine":
                    metric_value = calc_cosine(antecedent, consequent)
                
                # Add rule if metric value exceeds threshold
                if metric_value >= metric_threshold:
                    rules.append((set(antecedent), set(consequent), metric_value))
    
    return rules