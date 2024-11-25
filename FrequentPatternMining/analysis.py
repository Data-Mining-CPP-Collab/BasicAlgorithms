import patterns
import pandas as pd
import numpy as np
import testcases

# ---------------------
# Helper Functions
# ---------------------

def classify_column_into_categories(df, column):
    """
    Converts a numeric continuous column into classifier categories ('lowest', 'low', 'medium', 'high', 'highest').
    """
    categories = ['lowest', 'low', 'medium', 'high', 'highest']
    quantiles = df[column].quantile([0.2, 0.4, 0.6, 0.8]).values

    def classify(value):
        if value <= quantiles[0]:
            return 'lowest'
        elif value <= quantiles[1]:
            return 'low'
        elif value <= quantiles[2]:
            return 'medium'
        elif value <= quantiles[3]:
            return 'high'
        else:
            return 'highest'

    df[f"{column}_classified"] = df[column].apply(classify)
    return df

def prepare_data_for_apriori(df: pd.DataFrame, columns: list) -> list:
    """
    Filters the DataFrame based on the specified columns and prepares the data 
    for apriori and frequent itemset algorithms.
    """
    invalid_columns = [col for col in columns if col not in df.columns]
    if invalid_columns:
        raise ValueError(f"Invalid column names: {', '.join(invalid_columns)}")
    return df[columns].map(str).values.tolist()

def append_column_name_to_entries(df, column):
    """
    Appends column names to entries for clarity in apriori analysis.
    """
    df[f"{column}_labeled"] = df[column].astype(str) + f" ({column})"
    return df

def run_apriori_with_rules(
    df: pd.DataFrame,
    columns: list,
    metric: str,
    threshold_support: float,
    threshold_confidence: float,
):
    """
    Helper function to run apriori and generate association rules.
    """
    print(f"Running apriori on columns: {columns}")
    print()

    transaction_data = prepare_data_for_apriori(df, columns)
    frequent_itemsets = patterns.apriori(transaction_data, threshold=threshold_support)
    rules = patterns.association_rules(transaction_data, frequent_itemsets, metric, threshold_confidence)

    print(f"Association Rules (Metric: {metric}, Threshold: {threshold_confidence}):")
    testcases.show_rules(rules)
    testcases.show_itemsets(frequent_itemsets)

# ---------------------
# Main Analysis
# ---------------------

# Load and clean the data
LA_data_cleaned = pd.read_csv('LA_data_cleanedOCTOBER.csv')

# 1. Agent Name with City
print("Agent Name with City")
run_apriori_with_rules(
    df=LA_data_cleaned,
    columns=["agent_name", "city"],
    metric="confidence",
    threshold_support=0.001,
    threshold_confidence=0.1
)

# 2. Sqft with City
print("Sqft with City")
LA_data_cleaned = classify_column_into_categories(LA_data_cleaned, "sqft")
run_apriori_with_rules(
    df=LA_data_cleaned,
    columns=["sqft_classified", "city"],
    metric="confidence",
    threshold_support=0.01,
    threshold_confidence=0.01
)

# 3. City with School District
print("City with School District")
run_apriori_with_rules(
    df=LA_data_cleaned,
    columns=["city", "nearby_schools"],
    metric="confidence",
    threshold_support=0.01,
    threshold_confidence=0.01
)

# 4. Bath, Bed, with Style
print("Bath, Bed, with Style")
LA_data_cleaned = append_column_name_to_entries(LA_data_cleaned, "Total Baths")
LA_data_cleaned = append_column_name_to_entries(LA_data_cleaned, "beds")
run_apriori_with_rules(
    df=LA_data_cleaned,
    columns=["beds_labeled", "Total Baths_labeled", "style"],
    metric="confidence",
    threshold_support=0.1,
    threshold_confidence=0.1
)
