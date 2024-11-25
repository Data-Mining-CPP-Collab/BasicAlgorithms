import patterns
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import testcases

LA_data_cleaned = pd.read_csv('LA_data_cleanedOCTOBER.csv') 

def classify_column_into_categories(df, column):
    """
    Converts a numeric continuous column into classifier categories ('lowest', 'low', 'medium', 'high', 'highest').
    """

    categories = ['lowest', 'low', 'medium', 'high', 'highest']
    
    # Compute quantiles (20th, 40th, 60th, 80th percentiles)
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
    
    # Apply the classification function to the column
    classified_column = df[column].apply(classify)
    
    # Return the DataFrame with the new column
    df[f"{column}_classified"] = classified_column
    return df

def prepare_data_for_apriori(df: pd.DataFrame, columns: list) -> list:
    """
    Filters the DataFrame based on the specified columns and prepares the data 
    for apriori and frequent itemset algorithms.
    Returns:
    - list: A list of lists, where each inner list represents a transaction.
    """
    # Validate input columns
    invalid_columns = [col for col in columns if col not in df.columns]
    if invalid_columns:
        raise ValueError(f"Invalid column names: {', '.join(invalid_columns)}")

    # Filter the DataFrame to include only specified columns
    filtered_df = df[columns]

    # Convert the DataFrame to a list of lists
    transactions = filtered_df.applymap(str).values.tolist()

    return transactions

def append_column_name_to_entries(df, column):
    
    df[f"{column}_labeled"] = df[column].astype(str) + f" ({column})"
    return df

# ---------------------

print("agent name with city")
print()
agentAndCity = prepare_data_for_apriori(LA_data_cleaned, ["agent_name", "city"])

common = patterns.apriori(agentAndCity, 0.001)
rules = patterns.association_rules(agentAndCity, common, "confidence", 0.1)
print("Should find many association rules")
testcases.show_rules(rules)
testcases.show_itemsets(common)

# ----------------------

print("sqft with city")
print()
LA_data_cleaned = classify_column_into_categories(LA_data_cleaned, "sqft")
#previous line will create new column called sqft_classified

sqftAndCity = prepare_data_for_apriori(LA_data_cleaned, ["sqft_classified", "city"])

common = patterns.apriori(sqftAndCity, 0.01)
rules = patterns.association_rules(sqftAndCity, common, "confidence", 0.01)
print("Should find many association rules")
testcases.show_rules(rules)

# ----------------------

print("city with school district")
print()
#previous line will create new column called sqft_classified

cityAndSchool = prepare_data_for_apriori(LA_data_cleaned, ["city", "nearby_schools"])

common = patterns.apriori(cityAndSchool, 0.01)
rules = patterns.association_rules(cityAndSchool, common, "confidence", 0.01)
print("Should find many association rules")
testcases.show_rules(rules)

# ----------------------

LA_data_cleaned = append_column_name_to_entries(LA_data_cleaned, "Total Baths")
LA_data_cleaned = append_column_name_to_entries(LA_data_cleaned, "beds")


print("bath, bed, with style")
print()
#previous line will create new column called sqft_classified

bathBedStyle = prepare_data_for_apriori(LA_data_cleaned, ["beds_labeled", "Total Baths_labeled", "style"])

common = patterns.apriori(bathBedStyle, 0.1)
rules = patterns.association_rules(bathBedStyle, common, "confidence", 0.1)
print("Should find many association rules")
testcases.show_rules(rules)