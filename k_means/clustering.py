import random
import math
import numpy as np

# DO NOT CHANGE THE FOLLOWING LINE
def kmeans(data, k, columns, centers=None, n=None, eps=None):
# DO NOT CHANGE THE PRECEDING LINE
# This function has to return a list of k cluster centers (lists of floats of the same length as columns)
    # Define the maximum number of iterations and minimum convergence threshold
    if n is None:
        n = 100
    if eps is None:
        eps = 1e-4
    
    # Step 1: Extract columns used for clustering
    data = data[columns].values  # Extract only the columns used for clustering
    
    # Step 2: Initialize centroids
    if centers is None:
        # Randomly select k points from the data as initial centroids
        random_indices = np.random.choice(len(data), size=k, replace=False)
        centers = data[random_indices]
    else:
        centers = np.array(centers)
    
    # Step 3: Iteration until convergence or max iterations
    for _ in range(n):
        # Step 3a: Assign points to the nearest centroid
        # Calculate the distance between each point and each centroid using Eculidean method
        distances = np.sqrt(((data - centers[:, np.newaxis])**2).sum(axis=2))
        # Find the index of the closest centroid for each point
        cluster_labels = np.argmin(distances, axis=0)
        
        # Step 3b: Recalculate centroids
        new_centers = np.array([data[cluster_labels == i].mean(axis=0) for i in range(k)])
        
        # Step 3c: Check for convergence (stop if centers don't change much)
        if np.all(np.abs(new_centers - centers) < eps):
            break
        
        centers = new_centers
    
    # Convert centers to lists (based on the requirement)
    center_lists = [center.tolist() for center in centers]

    # Return the list of cluster centers
    return center_lists

# Distance function to calculate distance between a center and a data instance
def dist(center, instance, columns):
    # Extract the values for the given columns from the instance
    instance_vals = [instance[col] for col in columns]
    
    # Calculate Euclidean distance between center and instance values
    distance = np.sqrt(sum((c - i) ** 2 for c, i in zip(center, instance_vals)))
    
    return distance

# DO NOT CHANGE THE FOLLOWING LINE
def dbscan(data, columns, eps, min_samples):
# DO NOT CHANGE THE PRECEDING LINE
    # This function has to return a list of cluster centers (lists of floats of the same length as columns)
    pass
    
# DO NOT CHANGE THE FOLLOWING LINE
def kmedoids(data, k, distance, centers=None, n=None, eps=None):
# DO NOT CHANGE THE PRECEDING LINE
    # This function has to return a list of k cluster centroids (data instances!)
    pass

