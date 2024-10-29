import random
import numpy as np
import pandas as pd
from collections import deque
import math

def euclidean_distance(point1, point2):
  ''' each input is a list or array'''
  distance = 0
  for i in range(len(point1)):
    distance += (point1[i] - point2[i]) ** 2
  return math.sqrt(distance)
# DO NOT CHANGE THE FOLLOWING LINE
def kmedoids(data, k, distance, centers=None, n=None, eps=None):
# DO NOT CHANGE THE PRECEDING LINE
    """
    Perform k-medoids clustering on a dataset.

    Parameters:
    data (list of dicts or np.ndarray): The dataset to cluster.
    k (int): The number of clusters.
    distance (function): A function that calculates the distance between two data points.
    centers (list of data instances, optional): Initial medoid points.
    n (int, optional): Maximum number of iterations.
    eps (float, optional): Minimum change in medoids to declare convergence.

    Returns:
    list: A list of k medoid data points.
    """
    # Step 1: Initialize parameters
    if n is None:
        n = 100  # Default to 100 iterations if not specified
    if eps is None:
        eps = 1e-4  # Small value to check convergence
        
    if centers is None:
        # Step 2: Randomly select initial medoids if no centers are provided
        centers = random.sample(data, k)
    
    # Function to calculate total cost (sum of distances within clusters)
    def calculate_total_cost(centers, data, distance):
        total_cost = 0
        for d in data:
            mind = distance(centers[0], d)
            for center in centers[1:]:
                mind = min(mind, distance(center, d))
            total_cost += mind
        return total_cost

    # Step 3: Perform the K-Medoids algorithm
    for _ in range(n):
        clusters = [[] for _ in range(k)]  # Create empty lists for each cluster
        
        # Step 3a: Assign each data point to the nearest medoid
        for d in data:
            mind = distance(centers[0], d)
            idx = 0
            for i, center in enumerate(centers[1:], 1):
                dist_to_center = distance(center, d)
                if dist_to_center < mind:
                    mind = dist_to_center
                    idx = i

            # had to fix this code here correctly such that things work with mroe steps
            if 0 <= idx < k: 
                if len(clusters[idx]) == 0:
                    clusters[idx] = [d]
                else:
                    clusters[idx].append(d)
        
        #try to swap medoids with non-medoids and check for improvements - looking for improvement since this algo does not guarantee
        new_centers = []
        for i, cluster in enumerate(clusters):
            current_medoid = centers[i]
            best_medoid = current_medoid
            best_cost = calculate_total_cost(centers, data, distance)
            
            for candidate in cluster:
                new_centers = centers.copy()
                new_centers[i] = candidate
                new_cost = calculate_total_cost(new_centers, data, distance)
                
                if new_cost < best_cost:  # Swap
                    best_cost = new_cost
                    best_medoid = candidate
            
            new_centers.append(best_medoid)

        # Step 3c: Check if medoids have converged
        if all(distance(new, old) < eps for new, old in zip(new_centers, centers)):
            break
        
        centers = new_centers
    
    return centers



# DO NOT CHANGE THE FOLLOWING LINE
def kmeans(data, k, columns, centers=None, n=None, eps=None):
    # DO NOT CHANGE THE PRECEDING LINE
    # This function has to return a list of k cluster centers 
    
    if n is None:
        n = 100
    if eps is None:
        eps = 1e-4
    
    # Ensure data is a DataFrame 
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    data_for_kmeans = data.iloc[:, columns].values  # Extract only the specified 

    
    # Initialize centroids
    if centers is None:
        # Randomly select k points from the data as initial
        random_indices = np.random.choice(len(data_for_kmeans), size=k, replace=False)
        centers = data_for_kmeans[random_indices]
    else:
        centers = np.array(centers)
    
    # Iteration until convergence or max 
    for _ in range(n):
        # Assign points to the nearest centroid
        cluster_labels = []
        for point in data_for_kmeans:
            # Calculate distance to each centroid from the current point and assign point to the closest centroid
            distances = [euclidean_distance(point, center) for center in centers]
            closest_centroid = np.argmin(distances)
            cluster_labels.append(closest_centroid)
        cluster_labels = np.array(cluster_labels)
        
        # recalculate centroids
        new_centers = np.array([data_for_kmeans[cluster_labels == i].mean(axis=0) for i in range(k)])
        
        # Check for convergence (stop if centers don't change much)
        if np.all(np.abs(new_centers - centers) < eps):
            break
        
        centers = new_centers
    
    center_lists = [center.tolist() for center in centers]

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
    def region_query(data, point_idx, eps):
        """Find all points in the dataset within `eps` distance from the point at `point_idx`."""
        neighbors = []
        for i in range(len(data)):
            if euclidean_distance(data[point_idx], data[i]) < eps:
                neighbors.append(i)
        return neighbors

    def expand_cluster(data, labels, point_idx, neighbors, cluster_id, eps, min_samples):
        """Expand the cluster by finding density-reachable points."""
        labels[point_idx] = cluster_id 

        queue = deque(neighbors)

        while queue:
            neighbor_idx = queue.popleft()

            if labels[neighbor_idx] == -1: 
                labels[neighbor_idx] = cluster_id
            if labels[neighbor_idx] != 0:  # If it's already classified, skip
                continue

            labels[neighbor_idx] = cluster_id  # Assign cluster ID

            # Check the neighbors of this point
            new_neighbors = region_query(data, neighbor_idx, eps)
            if len(new_neighbors) >= min_samples:  # Only expand if it's a core point
                queue.extend(new_neighbors)

    if isinstance(data, pd.DataFrame):  
        new_df = data.copy()
        data = data[columns].to_numpy()

    labels = np.zeros(data.shape[0])  # 0 means unclassified, -1 means noise
    cluster_id = 0  # Cluster ID counter

    for i in range(len(data)):
        if labels[i] != 0:  # Skip if already classified
            continue

        neighbors = region_query(data, i, eps)
        if len(neighbors) < min_samples:
            labels[i] = -1  
        else:
            cluster_id += 1
            expand_cluster(data, labels, i, neighbors, cluster_id, eps, min_samples)

    

    labels_series = pd.Series(labels, name='labels')


    merged_df = pd.concat([new_df, labels_series], axis=1)


    return merged_df
    
# the following has the input use centers 
def kmeanswithlabels(data, k, columns, centers=None, n=None, eps=None):
    # DO NOT CHANGE THE PRECEDING LINE
    # This function has to return a list of k cluster centers (lists of floats of the same length as columns)
    
    # Define the maximum number of iterations and minimum convergence threshold
    if n is None:
        n = 100
    if eps is None:
        eps = 1e-4
    
    # Step 1: Extract columns used for clustering
    # Ensure data is a DataFrame and extract only the columns used for clustering
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, columns=columns)
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
        cluster_labels = []
        for point in data:
            # Calculate distance to each centroid from the current point and assign point to the closest centroid
            distances = [euclidean_distance(point, center) for center in centers]
            closest_centroid = np.argmin(distances)
            cluster_labels.append(closest_centroid)
        cluster_labels = np.array(cluster_labels)
        
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
