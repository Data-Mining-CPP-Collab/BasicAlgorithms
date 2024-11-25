import math
import matplotlib.pyplot as plt
import numpy as np
import random
import patterns
import pandas
import sys

random.seed(0)

APRIORI_TESTS = ["Find common letters in words (low threshold)", "Find common letters in words (high threshold)", "Find association rules for letters (low threshold)", "Find association rules for letters (high threshold)", "Compare association rule metrics on letters", "Find patterns in data set", "Find association rules in data set"]

def show_itemsets(itemsets):
    for itemset,support in itemsets:
        print(",".join(itemset), "%.2f"%(support))
        
def show_rules(rules):
    for condition,effect,metric in rules:
        print(",".join(condition), "=>", ",".join(effect), "%.2f"%(metric))

def write_itemsets_to_csv(itemsets, filename):
    """
    Displays and writes frequent itemsets to a CSV file.
    Args:
        itemsets: List of tuples, where each tuple contains a set of items and its support value.
        filename: Name of the CSV file to write to.
    """
    # Display the itemsets using show_itemsets
    print("Displaying itemsets:")
    show_itemsets(itemsets)

    # Convert itemsets to a Pandas DataFrame
    df = pandas.DataFrame([(",".join(itemset), support) for itemset, support in itemsets], 
                      columns=["Itemset", "Support"])

    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)
    print(f"Results saved to '{filename}'")
            
def apriori_testcase(data, letters, n):
    print("running test:", APRIORI_TESTS[n])
    if n == 0:
        # Test with dummy data
        common = patterns.apriori(letters, 0.1)
        
        # Test with supermarket transactions data
        # common = patterns.apriori(data, 0.1)
        
        print("This should find a few frequent itemsets with many items (letters)")
        # show_itemsets(common)
        # Display and save itemsets using the helper function
        write_itemsets_to_csv(common, "./FrequentPatternMining/test_cases_dummy_data/test_case_low_threshold.csv")
        # write_itemsets_to_csv(common, "./FrequentPatternMining/test_cases_supermarket/test_case_low_threshold.csv")
    elif n == 1:
        # Test with dummy data
        common = patterns.apriori(letters, 0.6)
        
        # Test with supermarket transactions data
        # common = patterns.apriori(data, 0.6)
        
        print("This should find a frequent itemset with few items (letters)")
        # show_itemsets(common)
        # Display and save itemsets using the helper function
        write_itemsets_to_csv(common, "./FrequentPatternMining/test_cases_dummy_data/test_case_high_threshold.csv")
        # write_itemsets_to_csv(common, "./FrequentPatternMining/test_cases_supermarket/test_case_high_threshold.csv")
    elif n == 2:
        common = patterns.apriori(letters, 0.1)
        rules = patterns.association_rules(letters, common, "all", 0.3)
        print("Should find many association rules")
        show_rules(rules)
    elif n == 3:
        common = patterns.apriori(letters, 0.5)
        rules = patterns.association_rules(letters, common, "all", 0.8)
        print("Should find very few association rules")
        show_rules(rules)
    elif n == 4:
        common = patterns.apriori(letters, 0.3)
        print("Will find several association rules, most for 'max', least/none for 'all'")
        for metric in ["lift", "all", "max", "kulczynski", "cosine"]:
            rules = patterns.association_rules(letters, common, metric, 0.71)
            print(metric + ": ")
            show_rules(rules)
            print()
    elif n == 5:
        common = patterns.apriori(data, 0.35)
        print("Should find several frequent itemsets with 2 items each")
        show_itemsets(common)
        #write_itemsets_to_csv(common, "./FrequentPatternMining/test_cases_dummy_data/test_case_find_patterns.csv")
        # write_itemsets_to_csv(common, "./FrequentPatternMining/test_cases_supermarket/test_case_find_patterns.csv")
    elif n == 6:
        common = patterns.apriori(data, 0.35)
        print("'all' may not return any rules, 'max' should return the most")
        for metric in ["lift", "all", "max", "kulczynski", "cosine"]:
            rules = patterns.association_rules(data, common, metric, 0.7)
            print(metric + ": ")
            show_rules(rules)
            print()
        
def to_set(row):
    result = set()
    for x in ["x1", "x2", "x3", "x4"]:
        if row[x] > 0.5:
            result.add(x)
    for c in ["cat1", "cat2", "cat3", "cat4"]:
        result.add(c + row[c])
    for b in ["bin1", "bin2", "bin3", "bin4", "bin5"]:
        if row[b] == "true":
            result.add(b)
    return result

# Convert each row of items into a set
def to_set2(row):
    # Convert the row of items into a set
    return {item for item in row if pandas.notna(item)}
    
def main(auto=False, steps=[]):
    words = ["loquacious", "insidious", "ferocious", "plausible", "atrocious",
                             "thematic", "vindicative", "automated", "pernicious", "advantageous",
                             "ambitious", "suspicious", "contentious", "curious", "guarded", "elusive",
                             "thousand", "approach", "intrusion", "suddenly", "obscure", "island", "ionic",
                             "oust", "obstinate", "foiled", "oily", "spoilers"]
    letters = list(map(set, words))
    # print(letters)
    
    # ----- Test with dummy data -----
    df = pandas.read_csv("./FrequentPatternMining/testdata.csv")
    
    # Convert each row to a set of items in the file testdata.csv
    data = [to_set(item) for (idx,item) in df.iterrows()]
    
    # ----- Test with supermarket data -----
    # df = pandas.read_csv("./FrequentPatternMining/supermarket_transactions.csv")
    
    # Convert each row to a set of items in the file supermartket_transactions.csv
    # data = [to_set2(item) for _, item in df.iterrows()]
    
    # print(data)
    if auto:
        for i,t in enumerate(APRIORI_TESTS):
            print("-"*80)
            apriori_testcase(data, letters, i)
        return
    tests = APRIORI_TESTS
    while True:
        print("Which test case do you want to run?")
        for i,t in enumerate(tests):
            print(f"   {i} {t}")
        print("   d print word data set")
        print("   q exit")
        if steps:
            x = steps[0]
            del steps[0]
        else:
            x = input("> ")
        if x in [str(i) for i,_ in enumerate(tests)]:
            apriori_testcase(data, letters, int(x))
        elif x == "d":
            print("Word list:\n  " + "\n  ".join(words))
        elif x == "q":
            print("Bye")
            sys.exit(0)
        else:
            print("Please select a test case, r or q")
        print()
    
    
    
if __name__ == "__main__":
    if "--help" in sys.argv:
        print("Usage: testcases.py [--auto] steps")
        print("   --auto runs the tests automatically")
        print("   <steps> is a sequence of inputs that are passed to the menu before it accepts manual input.")
        print("           This allows you to run e.g. 'python testcases.py 12q' to run test cases 1 and 2")
        print("           in sequence, followed by q(uit)")
        print("           Essentially, this allows you to repeatedly run any test/combination of tests without")
        print("           having to navigate the menu every time.")
        sys.exit(0)
    else:
        main("--auto" in sys.argv, 
             list("".join([arg for arg in sys.argv[1:] if "--" not in arg])))
        