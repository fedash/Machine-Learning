import scipy.io
import random
import numpy as np
import pandas as pd
from sklearn import metrics
from scipy import stats
mat = scipy.io.loadmat('mnist_10digits.mat')
random.seed(3680)

# -----------------------
# QUESTION 4.1
# -----------------------

k=10
#cluster  data 0-9
ytrain = mat['ytrain']

# data 0-255: standardize first
xtrain_standardized = mat['xtrain']/255



# ---------- Implement kmeans (L1 or L2 norm) ----------
def kmeans_purity(k, pixels, original_clusters, d_metric='euclidean'):
    """
    Implements K-means with random centroid initialization.
    ---------------------
            Input
    ---------------------
    k: The number of desired clusters
    pixels: A standardized 2D array of shape (d, m*n) with d digits of m*n pixel size each, such as 'xtrain' or         'xtest'
    original_clusters: An array with class assignments for each digit ('pixels'), such as 'ytrain' or 'ytest'
    d_metric: A distance metric to use, such as 'euclidian' or 'manhattan'
    ---------------------
            Output
    ---------------------
    original_c: A nested list: for each point in each cluster from the result of kmeans it contains the original
                cluster the point was assigned to. It is a list that can be further used for purity estimation.
    """
    # 0. Flatten the original cluster assignment array to 1D
    original_clusters = original_clusters.flatten()

    # 1. Initialize centroids
    initial = random.sample(range(pixels.shape[0]), k)
    centroids = pixels[initial]

    # 2. Assign points & recalculate centers
    max_iters = 500
    iters = 1

    while (iters <= max_iters):

        # 2.1 Calculate L2 distance from each datapoint to each centroid
        distance = metrics.pairwise_distances(pixels, centroids, metric=d_metric)

        # 2.2 Assign each point to the closest cluster
        closest = np.argmin(distance, axis=1)

        # 2.3 Calculate new centroids
        new_centroids = [] #as a list to change in case of empty clusters

        for c in range(k):

            # 2.3.1 Find the points that belong to cluster c
            points_in = pixels[closest == c]

            # 2.3.2 If c is not empty, set the new centroid as the mean or median of these points (depending on the metric)
            if len(points_in) != 0:
                if d_metric=='euclidean':
                    new_centroids.append(np.mean(points_in, axis=0))
                elif d_metric=='manhattan':
                    new_centroids.append(np.median(points_in, axis=0))

        # 2.4 Check if the algorithm has converged
        if np.allclose(new_centroids,centroids):
            break

        # 2.5 For next iteration, k is the number of non-empty clusters
        #     If no convergence, update centroids and keep iterating
        k = len(new_centroids)
        centroids = np.array(new_centroids)
        iters += 1

    # 3. Assign each data point to one of the final updated nearest centroids
    assigned_c = np.argmin(distance, axis=1)

    # 4. Nested list with original clusters for the data points in each of the resulting k clusters
    #    For each point in each cluster from kmeans, get the original cluster it was assigned to
    original_c  = [original_clusters[assigned_c == cl] for cl in range(k)]
    return original_c



# ---------- Calculate purity for each cluster ----------
def get_purity(original_c):
    # 1. Initiate empty lists to store purity scores, mode, mode counts and # of points for each cluster c
    purity = []
    modes = []
    mode_count = []
    cluster_size = []

    #2. Iterate through the lists with assignments for each cluster from kmeans
    for c in original_c:

        # 2.1 Get the mode (most common value) & the count of its value in the list
        c_mode, c_count = stats.mode(c)

        # 2.2 Purity = count of instances=mode in the cluster / # of data points in the cluster
        p = c_count[0] / len(c)

        # 2.3 Store purity, mode value, mode count and cluster size for cluster c
        purity.append(p)
        modes.append(c_mode[0])
        mode_count.append(c_count[0])
        cluster_size.append(len(c))

    # 3. Store purity scores for each cluster in a data frame
    d = {'Cluster #': range(len(original_c)),
         'Mode': modes,
         'Mode Count': mode_count,
         'Cluster Size': cluster_size,
         'Purity': purity}
    purity_df = pd.DataFrame(d)

    return purity, purity_df



# ---------- Get the results using Euclidean distance ----------
original_c_euclidean = kmeans_purity(k, xtrain_standardized, ytrain, d_metric='euclidean')
pur_e, purdf_e = get_purity(original_c_euclidean)
print(purdf_e)
w_pur_e = np.sum(purdf_e['Purity'] * purdf_e['Cluster Size']) / np.sum(purdf_e['Cluster Size'])
print(f'Weighted average purity, L2-norm: {w_pur_e}')



# -----------------------
# QUESTION 4.2
# -----------------------

# ---------- Get the results using Manhattan distance ----------
original_c_manhattan = kmeans_purity(k, xtrain_standardized, ytrain, d_metric='manhattan')
pur_m, purdf_m = get_purity(original_c_manhattan)
print(purdf_m)
w_pur_m = np.sum(purdf_m['Purity'] * purdf_m['Cluster Size']) / np.sum(purdf_m['Cluster Size'])
print(f'Weighted average purity, L1-norm: {w_pur_m}')