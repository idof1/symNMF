import numpy as np
import math
import sys
import symnmfmodule  # C module for SymNMF

# Original K-means functions from HW1

def read_vectors_from_txt(filename, delimiter=","):
    # Reads data from file into a numpy array
    vectors = []
    with open(filename, 'r') as f:
        for line in f:
            # Split the line based on the delimiter and convert each element to float
            vector = [float(x) for x in line.strip().split(delimiter)]
            vectors.append(vector)
    return vectors


def input_requirements(input_data, k, iteration='200'):
    # Checks input requirements, coped from HW1, even though unnecessary in the project
    # Under Section 2.9.2
    vectors = read_vectors_from_txt(input_data)

    # Check for k's validity - Unnecassary
    valid = False
    if k.isdigit():
        k = int(k)
        if 1 < k < len(vectors):
            valid = True
    if not valid:
        print("An Error Has Occurred")
        return vectors, k, iteration, valid

    # Check for iteration's validity
    valid = False
    if iteration.isdigit():
        iteration = int(iteration)
        if 1 < iteration < 1000:
            valid = True
    if not valid:
        print("An Error Has Occurred")
        return vectors, k, iteration, valid

    #If both are valid, return all
    return vectors, k, iteration, valid


def k_means(vectors, k, iter, eps=0.0001):
    #K_means as implemented in HW1.
    centroids = []
    clusters = None
    for i in range(k):
        centroids.append(vectors[i])
    for i in range(iter):
        clusters = [[] for j in range(k)]
        for v in vectors:
            min_d, min_ind = euclidean_distance(v, centroids[0]), 0
            for j in range(k):
                d = euclidean_distance(v, centroids[j])
                if d < min_d:
                    min_d, min_ind = d, j
            clusters[min_ind].append(v)

        new_centroids = []
        for j in range(k):
            if clusters[j]:
                new_centroids.append(update_centroid(clusters[j]))
            else:
                new_centroids.append(centroids[j])

        flag = False
        for j in range(k):
            if euclidean_distance(new_centroids[j], centroids[j]) > eps:
                flag = True
        if not flag:
            return clusters, centroids
        centroids = new_centroids
    return clusters, centroids


def euclidean_distance(point1, point2):
    #Calculates euclidean distance between two points
    d = 0
    for i in range(len(point1)):
        d += (point1[i] - point2[i]) ** 2
    return math.sqrt(d)


def update_centroid(cluster):
    #Updates the centroid of a cluster
    vector_dim = len(cluster[0])
    s = [0 for i in range(vector_dim)]
    for ind in range(vector_dim):
        for v in cluster:
            s[ind] += v[ind]
        s[ind] = s[ind] / len(cluster)
    return s

# Manual silhouette score calculation function
def calculate_silhouette_score(X, labels, k):
    silhouette_scores = []

    for i, point in enumerate(X):
        # Find the cluster of the current point
        own_cluster = labels[i]
        
        # Calculate 'a' (intra-cluster distance)
        intra_distances = [euclidean_distance(point, X[j]) for j in range(len(X)) if labels[j] == own_cluster and i != j]
        a = np.mean(intra_distances) if len(intra_distances) != 0 else 0
        
        # Calculate 'b' (nearest-cluster distance)
        inter_distances = []
        for cluster in range(k):
            if cluster != own_cluster:
                cluster_distances = [euclidean_distance(point, X[j]) for j in range(len(X)) if labels[j] == cluster]
                if cluster_distances:
                    inter_distances.append(np.mean(cluster_distances))
        b = min(inter_distances) if len(intra_distances) != 0 else 0
        
        # Calculate silhouette score for the point
        silhouette = (b - a) / max(a, b) if max(a, b) != 0 else 0
        silhouette_scores.append(silhouette)
    
    return np.mean(silhouette_scores)

# Perform the analysis for both SymNMF and K-means
def analyze_clustering(k, file_name):
    # Load data from the file
    vectors = read_vectors_from_txt(file_name)
    iterations = 300
    X = np.array(vectors)

    # SymNMF Clustering
    np.random.seed(1234)
    # Call the C module to perform symnmf
    A = symnmfmodule.sym(X.tolist())
    D = symnmfmodule.ddg(A)
    W = symnmfmodule.norm(A, D)
    m = np.mean(W)
    H = np.random.uniform(0, 2 * np.sqrt(m / k), size=(X.shape[0], k))
    final_H = symnmfmodule.symnmf(W, H.tolist(), 300, 1e-4)
    
    # Assign clusters based on the maximum value in H (soft-to-hard clustering)
    symnmf_labels = np.argmax(final_H, axis=1)
    symnmf_score = calculate_silhouette_score(X, symnmf_labels, k)

    # K-means Clustering using original HW1 code
    clusters, centroids = k_means(vectors, k, iterations)
    kmeans_labels = np.array([label for label, cluster in enumerate(clusters) for _ in cluster])
    kmeans_points = np.array([_ for label, cluster in enumerate(clusters) for _ in cluster])
    kmeans_score = calculate_silhouette_score(kmeans_points, kmeans_labels, k)

    # Print the silhouette scores for both methods, formatted to 4 decimal places
    print(f"nmf: {symnmf_score:.4f}")
    print(f"kmeans: {kmeans_score:.4f}")

# Main function to parse command-line arguments
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("An Error Has Occured")
    
    else:
        k = int(sys.argv[1])
        file_name = sys.argv[2]

        analyze_clustering(k, file_name)
