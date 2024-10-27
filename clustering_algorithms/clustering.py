import random
import numpy as np
import pandas as pd
from collections import deque

def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt(np.sum((point1 - point2) ** 2))

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
            clusters[idx].append(d)
        
        # Step 3b: Try to swap medoids with non-medoids and check for improvements
        new_centers = []
        for i, cluster in enumerate(clusters):
            current_medoid = centers[i]
            best_medoid = current_medoid
            best_cost = calculate_total_cost(centers, data, distance)
            
            for candidate in cluster:
                new_centers = centers.copy()
                new_centers[i] = candidate
                new_cost = calculate_total_cost(new_centers, data, distance)
                
                if new_cost < best_cost:  # Swap if we find a better medoid
                    best_cost = new_cost
                    best_medoid = candidate
            
            new_centers.append(best_medoid)

        # Step 3c: Check if medoids have converged
        if all(distance(new, old) < eps for new, old in zip(new_centers, centers)):
            break
        
        centers = new_centers
    
    return centers

# Sample datasets for testing
# data1 = [
#     [1, 2], [2, 3], [3, 4], [10, 10], [11, 11], [12, 12]
# ]  # Expected: 2 clusters around [1, 2] and [10, 10] or similar

# data2 = [
#     [5, 5], [5, 6], [6, 5], [6, 6], [25, 25], [26, 26], [25, 26], [26, 25]
# ]  # Expected: 2 clusters around [5,5] and [25,25] or similar

# # Running k-medoids on data1 with k=2
# centers1 = kmedoids(data1, k=2, distance=euclidean_distance)
# print("Test Case 1 - Data1 Centers:", centers1)

# # Running k-medoids on data2 with k=2
# centers2 = kmedoids(data2, k=2, distance=euclidean_distance)
# print("Test Case 2 - Data2 Centers:", centers2)

# # Running k-medoids on data1 with k=3
# centers3 = kmedoids(data1, k=3, distance=euclidean_distance)
# print("Test Case 3 - Data1 Centers (k=3):", centers3)