import classification
import pandas as pd
import numpy as np
import testcases
import random
import time
import json

# convert columns into categorized ones like last time - DONE

# create method to do the 70 15 15 split - DONE

# separate y and x's 

# calculate performace 

# helper method like last time for modular output?
# classification testcase method 



# this is from profs code
def get_columns(rows, columns, single=False):
    if single:
        return [row[columns[0]] for row in rows]
    return [[row[c] for c in columns] for row in rows]

LA_data_cleaned = pd.read_csv('LA_data_cleanedOCTOBER.csv')

def classify_column_into_categories(df, column):
    """
    Converts a numeric continuous column into classifier categories with ranges.
    Handles both integer and float values appropriately.
    """
    # Calculate quantiles
    quantiles = df[column].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).values
    
    # Determine appropriate rounding based on the data type and magnitude
    if df[column].dtype in ['int64', 'float64']:
        # For price columns, round to nearest thousand if values are large
        if 'price' in column.lower() or 'value' in column.lower():
            quantiles = [round(q, -3) for q in quantiles]  # Round to thousands
        # For sqft, round to nearest 10
        elif 'sqft' in column.lower():
            quantiles = [round(q, -1) for q in quantiles]  # Round to tens
        # For other numeric columns, round to 2 decimal places
        else:
            quantiles = [round(q, 2) for q in quantiles]

    def classify(value):
        if pd.isna(value):  # Handle NaN values
            return f'unknown_{column}'
        elif value <= quantiles[1]:
            return f'lowest_{column}({quantiles[0]}-{quantiles[1]})'
        elif value <= quantiles[2]:
            return f'low_{column}({quantiles[1]}-{quantiles[2]})'
        elif value <= quantiles[3]:
            return f'medium_{column}({quantiles[2]}-{quantiles[3]})'
        elif value <= quantiles[4]:
            return f'high_{column}({quantiles[3]}-{quantiles[4]})'
        else:
            return f'highest_{column}({quantiles[4]}-{quantiles[5]})'

    # Create new column with range-based categories
    df[f"{column}_classified"] = df[column].apply(classify)
    
    # Store the ranges in the dataframe metadata (optional)
    if not hasattr(df, 'category_ranges'):
        df.category_ranges = {}
    
    df.category_ranges[column] = {
        f'lowest_{column}': f'({quantiles[0]}-{quantiles[1]})',
        f'low_{column}': f'({quantiles[1]}-{quantiles[2]})',
        f'medium_{column}': f'({quantiles[2]}-{quantiles[3]})',
        f'high_{column}': f'({quantiles[3]}-{quantiles[4]})',
        f'highest_{column}': f'({quantiles[4]}-{quantiles[5]})'
    }
    
    return df

def classify_multiple_columns(df, columns_to_classify):
    """
    Applies the classification with ranges to multiple columns.
    Prints out the range information for each column.
    """
    df_copy = df.copy()
    
    # Dictionary to store range information for all columns
    all_ranges = {}
    
    # commented out printing, wwe printed so we could see range outputs
    for column in columns_to_classify:
        if column in df.columns:
            #print(f"\nClassifying {column}:")
            df_copy = classify_column_into_categories(df_copy, column)
            
            # Print the ranges for this column
            if hasattr(df_copy, 'category_ranges') and column in df_copy.category_ranges:
                #print("Ranges:")
                #for category, range_val in df_copy.category_ranges[column].items():
                    #print(f"  {category}: {range_val}")
                all_ranges[column] = df_copy.category_ranges[column]
        #else:
            #print(f"Warning: Column '{column}' not found in dataframe")
    
    # Store all ranges in the dataframe for future reference
    df_copy.all_category_ranges = all_ranges
    
    return df_copy


# we create new columns
columns_to_classify = ['sqft', 'beds', 'year_built', 'days_on_mls',
       'list_price', 'sold_price', 'estimated_value',
       'lot_sqft', 'price_per_sqft', 'latitude',
       'longitude', 'hoa_fee']  # Replace with your column names
LA_data_cleaned_classified = classify_multiple_columns(LA_data_cleaned, columns_to_classify)

# we just created more columns in the padas df that are converted from continuous into quantiles

# the three way split 70 15 15

def split_dataset(df, training, testing, validation):
    """
    Splits the dataset into training, testing, and validation sets (70%, 15%, 15%).

    Parameters:
    - df: The input dataframe.
    - training: List to hold training data.
    - testing: List to hold testing data.
    - validation: List to hold validation data.

    Returns:
    - None (modifies the lists in-place).
    """
    random.seed(time.time())  # Seed randomness with the current time

    for i, row in df.iterrows():
        rand_value = random.random()
        if rand_value > 0.85:
            validation.append(row)
        elif rand_value > 0.7:
            testing.append(row)
        else:
            training.append(row)

# we will split the data into training, testing, and validation
training = []
testing = []
validation = []
split_dataset(LA_data_cleaned_classified, training, testing, validation)

# models dictionary
MODELS = {"Decision Tree": classification.DecisionTree, "Naive Bayes": testcases.NaiveBayes}

def calculate_performance(prefix, y, predy):
    # Calculate accuracy
    correct = sum(1 for p, v in zip(predy, y) if p == v)
    accuracy = correct / len(y)
    
    # Get unique classes
    classes = sorted(set(y))
    
    print(f"{prefix}Accuracy: {accuracy:.2f}")
    print("\nPer-class metrics:")
    
    # cls is like quantiles, or like the specific style of home
    for cls in classes:
        # True Positives
        tp = sum(1 for p, v in zip(predy, y) if p == cls and v == cls)
        
        # False Positives
        fp = sum(1 for p, v in zip(predy, y) if p == cls and v != cls)
        
        # False Negatives
        fn = sum(1 for p, v in zip(predy, y) if p != cls and v == cls)
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\nClass: {cls}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")

def classification_run(training, validation, columns, target, model="Decision Tree", max_depth=6):

    print()
    print('------------------------ TRAINING PREFORMANCE ---------------------------')
    print()
    # all training
    m = MODELS[model](max_depth=max_depth)
    tx = get_columns(training, columns)
    ty = get_columns(training, target, single=True)
    m.fit(tx, ty)
    predty = m.predict(tx)
    calculate_performance(model + " training ", ty, predty)

    print()
    print('------------------------ VALIDATION PREFORMANCE --------------------------')
    print()
    # validation
    vx = get_columns(validation, columns)
    vy = get_columns(validation, target, single=True)
    predvy = m.predict(vx)
    calculate_performance(model + " validation ", vy, predvy)

    #after we will do testing

    # also i am going to return m so we can test predictions with data of our own that we want
    return m

# these are all the columns, rememeber, many of them were made into quantiles and added to the df

"""
Index(['Unnamed: 0', 'property_url', 'style', 'street', 'city', 'zip_code',
       'beds', 'Total Baths', 'sqft', 'year_built', 'days_on_mls',
       'list_price', 'sold_price', 'last_sold_date', 'estimated_value',
       'new_construction', 'lot_sqft', 'price_per_sqft', 'latitude',
       'longitude', 'neighborhoods', 'stories', 'hoa_fee', 'parking_garage',
       'agent_name', 'nearby_schools'],
      dtype='object')
"""

""" --------------------
Column: sqft_classified
  Unique Values: ['high_sqft(1860.0-2520.0)' 'low_sqft(1140.0-1480.0)'
 'highest_sqft(2520.0-77860.0)' 'medium_sqft(1480.0-1860.0)'
 'lowest_sqft(400.0-1140.0)']
--------------------
Column: beds_classified
  Unique Values: ['high_beds(3.0-4.0)' 'lowest_beds(0.0-2.0)' 'highest_beds(4.0-26.0)'
 'low_beds(2.0-3.0)']
--------------------
Column: year_built_classified
  Unique Values: ['low_year_built(1935.0-1951.6)' 'high_year_built(1963.0-1984.0)'
 'medium_year_built(1951.6-1963.0)' 'highest_year_built(1984.0-2024.0)'
 'lowest_year_built(1871.0-1935.0)']
--------------------
Column: days_on_mls_classified
  Unique Values: ['highest_days_on_mls(98.0-954.0)' 'medium_days_on_mls(48.0-67.0)'
 'high_days_on_mls(67.0-98.0)' 'low_days_on_mls(36.0-48.0)'
 'lowest_days_on_mls(0.0-36.0)']
--------------------
Column: list_price_classified
  Unique Values: ['medium_list_price(880000.0-1195000.0)'
 'lowest_list_price(85000.0-699000.0)'
 'high_list_price(1195000.0-1699000.0)'
 'low_list_price(699000.0-880000.0)'
 'highest_list_price(1699000.0-26995000.0)']
--------------------
Column: sold_price_classified
  Unique Values: ['medium_sold_price(876000.0-1194000.0)'
 'lowest_sold_price(85000.0-700000.0)'
 'high_sold_price(1194000.0-1695000.0)'
 'highest_sold_price(1695000.0-24000000.0)'
 'low_sold_price(700000.0-876000.0)']
--------------------
Column: estimated_value_classified
  Unique Values: ['medium_estimated_value(903000.0-1223000.0)'
 'lowest_estimated_value(85000.0-714000.0)'
 'high_estimated_value(1223000.0-1745000.0)'
 'highest_estimated_value(1745000.0-20585000.0)'
 'low_estimated_value(714000.0-903000.0)']
--------------------
Column: lot_sqft_classified
  Unique Values: ['low_lot_sqft(5520.0-6750.0)' 'highest_lot_sqft(19590.0-787600.0)'
 'medium_lot_sqft(6750.0-8560.0)' 'lowest_lot_sqft(50.0-5520.0)'
 'high_lot_sqft(8560.0-19590.0)']
--------------------
Column: price_per_sqft_classified
  Unique Values: ['low_price_per_sqft(0.0-1000.0)' 'highest_price_per_sqft(1000.0-7000.0)']
--------------------
Column: latitude_classified
  Unique Values: ['lowest_latitude(33.71-34.0)' 'highest_latitude(34.21-34.33)'
 'high_latitude(34.15-34.21)' 'medium_latitude(34.06-34.15)'
 'low_latitude(34.0-34.06)']
--------------------
Column: longitude_classified
  Unique Values: ['highest_longitude(-118.29--118.13)' 'low_longitude(-118.5--118.43)'
 'lowest_longitude(-118.67--118.5)' 'high_longitude(-118.36--118.29)'
 'medium_longitude(-118.43--118.36)']
--------------------
Column: hoa_fee_classified
  Unique Values: ['lowest_hoa_fee(0.0-0.0)' 'high_hoa_fee(0.0-405.4)'
 'highest_hoa_fee(405.4-8693.0)']
--------------------
 """

def filter_properties(df, feature_values, feature_names, columns_to_return=None):
   
    try:
        # Convert feature_values to list if it's a numpy array
        if isinstance(feature_values, np.ndarray):
            feature_values = feature_values.flatten().tolist()
            
        # Create filter conditions
        conditions = True
        for feature, value in zip(feature_names, feature_values):
            conditions = conditions & (df[feature] == value)
            
        # Apply filters
        filtered_df = df[conditions]
        
        # Select specific columns if requested
        if columns_to_return:
            filtered_df = filtered_df[columns_to_return]
            
        print(f"Found {len(filtered_df)} matching properties")
        
        return filtered_df
        
    except KeyError as e:
        print(f"Error: Column not found in dataframe: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error occurred while filtering: {e}")
        return pd.DataFrame()

print()
##########################################################################################
print('----------- THE FOLLOWING IS THE HOUSEHUNTERS APPLICATION -----------------')
##########################################################################################
print()

features1 = ['sqft_classified', 'estimated_value_classified','beds_classified', 'year_built_classified']
yvariable1 = ['city']

m1 = classification_run(training, validation, features1, 
                                              yvariable1, 
                                              model="Decision Tree", 
                                              max_depth=3)
# this will output all the scores we want
print(json.dumps(m1.to_dict(), indent=4))

# Format the array as a 2D array expects 2D input
test_array1 = np.array([
                    'high_sqft(1860.0-2520.0)', 
                    'medium_estimated_value(903000.0-1223000.0)', 
                    'high_beds(3.0-4.0)',
                    'highest_year_built(1984.0-2024.0)']
                     ).reshape(1, -1)
# Get prediction
print()
print('------------------------ OUR PREDICTION FROM EXAMPLE INPUT --------------------------')
print()
print('the input was', test_array1)
prediction = m1.predict(test_array1)
print("Predicted city:", prediction[0])

print("Examples: ")
# Get matches with specific columns
matches = filter_properties(
    LA_data_cleaned_classified, 
    test_array1[0], 
    features1,
    ['property_url']
)

# Print results
print(matches.to_string())


print()
###################################################################################################
print('----------- THE FOLLOWING IS THE TIME ON MARKET ESTIMATE APPLICATION -----------------')
###################################################################################################
print()

features2 = ['sqft_classified', 'estimated_value_classified', 'beds_classified', 'city']
yvariable2 = ['days_on_mls_classified']

m2 = classification_run(training, validation, features2, 
                                              yvariable2, 
                                              model="Decision Tree",
                                              max_depth=2)
# this will output all the scores we want
print(json.dumps(m2.to_dict(), indent=4))

# Format the array as a 2D array expects 2D input
test_array2 = np.array([
                    'medium_sqft(1480.0-1860.0)', 
                    'medium_estimated_value(903000.0-1223000.0)', 
                    'high_beds(3.0-4.0)',
                    'North Hollywood'
                    ]
                     ).reshape(1, -1)

# Get prediction
print()
print('------------------------ OUR PREDICTION FROM EXAMPLE INPUT --------------------------')
print()
print('the input was', test_array2,)
prediction2 = m2.predict(test_array2)
print("Predicted time on market range:", prediction2[0])

matches = filter_properties(
    LA_data_cleaned_classified, 
    test_array2[0], 
    features2,
    ['property_url']
)

# Print results
print(matches.to_string())
