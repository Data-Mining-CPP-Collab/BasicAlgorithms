import clustering as cl
import pandas as pd
import numpy as np
import math


"""
To actually use clustering on your data sets, create a new python file analysis.py that contains the code for your analysis, 
and that import clustering . This way you will not accidentally break the other test cases as you do your analysis 
(of course, if you find a bug in your clustering algorithm in this step, you should fix it and rerun the other tests).


Once you have implemented K-Means and DBSCAN algorithm and tested it thoroughly, it is time to apply it to 
your chosen data set. First you will have to consider what you expect a clustering to mean, and which attributes 
may be useful to get it. Before you apply the clustering algorithm do not forget to normalize your attributes so 
that they are all within the same range! You can always try your prepared data with an arbitrary k to make sure 
everything is working properly. Note that if you have many categorical attributes that you can not easily convert 
to numbers, you might want to consider implementing k- medoids, as that can work with such attributes.


The next step is to determine how to evaluate clusters! Given what you expect your clusters to mean, 
come up with a way to verify that the clusters produced by your algorithm are “reasonable”. 
What would it mean for two elements to be in the same cluster? What if they are in different clusters? 
If you have any “strong” intuition whether any particular instances should be in the same cluster 
or not (e.g. “DotA 2” and “LoL” should be in the same cluster, but “Chess” should be in a different one, 
if you expect your clusters to have to do with game play/mechanics), they can make excellent test cases 
(but that should not be the only criterion you look at). A good evaluation criterion would be something 
you can evaluate programmatically, e.g. a number you calculate or a graph that shows the clusters.


To effectively assess the quality of your clusters, you can use both intrinsic and extrinsic 
evaluation measures that was taught in class.

· Intrinsic Evaluation Measures: These metrics assess the clustering results based on the data itself without reference to the ground truth. Common intrinsic measures include the silhouette score, which evaluates how similar an object is to its own cluster compared to other clusters, may be used.

· Extrinsic Evaluation Measures: In contrast, extrinsic measures require external information, such as ground truth labels, to evaluate the quality of clusters. Examples include adjusted Rand index (ARI) and normalized mutual information (NMI), which compare the clustering results to known classifications. These metrics can provide a clear indication of how well your clustering algorithm performed in relation to expected groupings.

If you are using extrinsic evaluation, the you may think about generating the ground truth by 
labeling the data objects manually if the ground truth is not available.




With a way to evaluate your clusters in hand, you are now ready to actually calculate clusters. 
To do so, you will need to find good values for the parameters. 
For the number of clusters you can start with an intuitive guess, 
or simply start with 1 and increase it one by one, and evaluate each result using your evaluation criterion until you are happy 
(it may happen that there is more than one “good” value; if this happens, be sure to mention it in your report). 
For the termination criteria, you may want to use both (with a relatively large n ), to prevent the algorithm from “oscillating”, 
but to stop once the clusters do not change anymore. It may be helpful to print the movement distance of 
the cluster centers in each iteration to get a better picture of what is happening.


Finally, once you have clusters that satisfy your criterion, you will need to think about how to present them. 
For this, consider what “defines” each cluster (this will likely not be something you can automate), 
and describe each cluster in these terms. Please do not include a complete list of instances for each cluster in your report 
(especially not if there are thousands such instances). If any individual cluster is small and interesting, 
you can list its element, though.


Note: It may happen that there is no “good” clustering. In this case, make sure to note which parameter 
settings you tried, and why none of them lead to a satisfactory outcome
"""

LA_data_cleaned = pd.read_csv('LA_data_cleanedOCTOBER.csv') 

#cleaning the data
numerical_df = LA_data_cleaned.select_dtypes(include=np.number)
numerical_df = numerical_df.drop('zip_code', axis=1, errors='ignore')
numerical_df['property_url'] = LA_data_cleaned['property_url']

columns_of_choice = ['price_per_sqft', 'list_price']

def euclidean_distance(point1, point2):
  ''' each input is a list or array'''
  distance = 0
  for i in range(len(point1)):
    distance += (point1[i] - point2[i]) ** 2
  return math.sqrt(distance)



#intrinsic measure - silhouette coefficient
def silhouette_coefficient(df):
    """
    Calculate the Silhouette Coefficient for each point in the dataset, ignoring outliers.
    """
    features = df.iloc[:, :-2]  # All columns except the last one
    labels = df.iloc[:, -1]     # The last column

    n = len(df)
    silhouette_scores = np.full(n, np.nan)  # Initialize silhouette scores as NaN for outliers

    unique_labels = np.unique(labels)

    # Iterate through all data points, excluding outliers
    for i in range(n):
        current_point = features.iloc[i].to_numpy()
        current_label = labels[i]

        if current_label == -1: # skip
            continue

        # Get points in the same cluster as the current point
        same_cluster_mask = (labels == current_label)
        same_cluster_points = features[same_cluster_mask].to_numpy()

        # Cohesion (a): Mean intra-cluster distance
        if len(same_cluster_points) > 1:
            a = np.mean([euclidean_distance(current_point, p) for p in same_cluster_points if not np.array_equal(p, current_point)])
        else:
            a = 0  # If it's the only point in the cluster, set a to 0

        # Separation (b): Mean distance to the nearest cluster
        b = np.inf
        for other_label in unique_labels:
            if other_label != current_label and other_label != -1:
                other_cluster_points = features[labels == other_label].to_numpy()
                mean_dist_to_other_cluster = np.mean([euclidean_distance(current_point, p) for p in other_cluster_points])
                b = min(b, mean_dist_to_other_cluster)

        silhouette_scores[i] = (b - a) / max(a, b)

    # Return the silhouette scores as a pandas Series
    return pd.Series(silhouette_scores, index=df.index)



def remove_outliers_iqr(df, column):
  """Removes outliers from a pandas DataFrame column using the IQR method."""
  Q1 = df[column].quantile(0.25)
  Q3 = df[column].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
  return df_filtered


for column in numerical_df.select_dtypes(include=np.number).columns:
  numerical_df = remove_outliers_iqr(numerical_df, column)

numerical_df



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

normalized_data = scaler.fit_transform(numerical_df[columns_of_choice])

normalized_df = pd.DataFrame(normalized_data, columns=columns_of_choice)
normalized_df['property_url'] = numerical_df['property_url'].values
normalized_df

withlabels = cl.dbscan(normalized_df, columns_of_choice, eps=0.05, min_samples=5)
withlabels