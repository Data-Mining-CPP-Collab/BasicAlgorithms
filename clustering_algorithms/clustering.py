import random
import math
import numpy as np
import pandas as pd
import sys
from collections import deque

# DO NOT CHANGE THE FOLLOWING LINE
def kmeans(data, k, columns, centers=None, n=None, eps=None):
# DO NOT CHANGE THE PRECEDING LINE
# This function has to return a list of k cluster centers (lists of floats of the same length as columns)

    # Distance function to calculate distance between a center and a data instance
    def dist(center, instance, columns):
        # Extract the values for the given columns from the instance
        instance_vals = [instance[col] for col in columns]
        
        # Calculate Euclidean distance between center and instance values
        distance = np.sqrt(sum((c - i) ** 2 for c, i in zip(center, instance_vals)))
        
        return distance

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


# DO NOT CHANGE THE FOLLOWING LINE
def dbscan(data, columns, eps, min_samples):
# DO NOT CHANGE THE PRECEDING LINE
    """
    Perform DBSCAN clustering on the specified columns of the dataset.

    Parameters:
    data (pd.DataFrame or np.ndarray): The dataset.
    columns (list): List of column names to use for clustering (if using DataFrame).
    eps (float): Maximum distance for two points to be considered neighbors.
    min_samples (int): The minimum number of samples in a neighborhood to form a core point.

    Returns:
    np.ndarray: Cluster labels for each data point (-1 indicates noise).
    """

    def euclidean_distance(point1, point2):
        """Calculate the Euclidean distance between two points."""
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def region_query(data, point_idx, eps):
        """Find all points in the dataset within `eps` distance from the point at `point_idx`."""
        neighbors = []
        for i in range(len(data)):
            if euclidean_distance(data[point_idx], data[i]) < eps:
                neighbors.append(i)
        return neighbors

    def expand_cluster(data, labels, point_idx, neighbors, cluster_id, eps, min_samples):
        """Expand the cluster by finding density-reachable points."""
        labels[point_idx] = cluster_id  # Assign the cluster label to the current point

        # Use a queue to explore all neighbors (directly reachable points)
        queue = deque(neighbors)

        while queue:
            neighbor_idx = queue.popleft()

            if labels[neighbor_idx] == -1:  # If it was marked as noise, now it's part of the cluster
                labels[neighbor_idx] = cluster_id
            if labels[neighbor_idx] != 0:  # If it's already classified, skip
                continue

            labels[neighbor_idx] = cluster_id  # Assign cluster ID

            # Check the neighbors of this point
            new_neighbors = region_query(data, neighbor_idx, eps)
            if len(new_neighbors) >= min_samples:  # Only expand if it's a core point
                queue.extend(new_neighbors)

    if isinstance(data, pd.DataFrame):  # Convert DataFrame to NumPy array if necessary
        data = data[columns].to_numpy()

    labels = np.zeros(data.shape[0])  # 0 means unclassified, -1 means noise
    cluster_id = 0  # Cluster ID counter

    for i in range(len(data)):
        if labels[i] != 0:  # Skip if already classified
            continue

        neighbors = region_query(data, i, eps)
        if len(neighbors) < min_samples:
            labels[i] = -1  # Mark as noise
        else:
            cluster_id += 1
            expand_cluster(data, labels, i, neighbors, cluster_id, eps, min_samples)

    return labels
    

# DO NOT CHANGE THE FOLLOWING LINE
def kmedoids(data, k, distance, centers=None, n=None, eps=None):
# DO NOT CHANGE THE PRECEDING LINE
    # This function has to return a list of k cluster centroids (data instances!)
    pass

