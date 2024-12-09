import classification
import pandas as pd
import numpy as np
import testcases
import random
import time

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

    # creates a new column called sqft_classified, for example
    df[f"{column}_classified"] = df[column].apply(classify)
    return df

# our data
LA_data_cleaned = pd.read_csv('LA_data_cleanedOCTOBER.csv')

def classify_multiple_columns(df, columns_to_classify):
    
    df_copy = df.copy()
    
    for column in columns_to_classify:
        if column in df.columns:
            df_copy = classify_column_into_categories(df_copy, column)
        else:
            print(f"Warning: Column '{column}' not found in dataframe")
    
    return df_copy

# we create new columns
columns_to_classify = ['sqft', 'beds', 'year_built', 'days_on_mls',
       'list_price', 'sold_price', 'estimated_value',
       'lot_sqft', 'price_per_sqft', 'latitude',
       'longitude', 'hoa_fee']  # Replace with your column names
LA_data_cleaned_classified = classify_multiple_columns(LA_data_cleaned, columns_to_classify)

### we just created more columns in the padas df that are converted from continuous into quantiles

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

def classification_run(training, validation, columns, target, model="Naive Bayes", testing=None):

    print()
    print('------------------------ TRAINING PREFORMANCE ---------------------------')
    print()
    # all training
    m = MODELS[model]()
    tx = get_columns(training, columns)
    ty = get_columns(training, target, single=True)
    m.fit(tx, ty)
    predty = m.predict(tx)
    calculate_performance(model + " training ", ty, predty)

    print()
    print('---------------------VALIDATION PREFORMANCE-----------------')
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

 # Index(['Unnamed: 0', 'property_url', 'style', 'street', 'city', 'zip_code',
 #      'beds', 'Total Baths', 'sqft', 'year_built', 'days_on_mls',
 #      'list_price', 'sold_price', 'last_sold_date', 'estimated_value',
 #      'new_construction', 'lot_sqft', 'price_per_sqft', 'latitude',
 #     'longitude', 'neighborhoods', 'stories', 'hoa_fee', 'parking_garage',
 #      'agent_name', 'nearby_schools'],
 #     dtype='object')

m1 = classification_run(training, validation, ['sqft_classified', 'estimated_value_classified',
                                           'beds_classified'], ['style'], model="Naive Bayes")
# this will output all the scores we want

# Format the array as a 2D array expects 2D input
test_array = np.array(['medium', 'medium', 'high']).reshape(1, -1)

# Get prediction
print()
print('---------our prediction from input---------')
print()
print('the input was', test_array, 'for the parameters', ['sqft_classified', 'estimated_value_classified',
                                           'beds_classified'])
prediction = m1.predict(test_array)
print("Predicted style:", prediction[0])