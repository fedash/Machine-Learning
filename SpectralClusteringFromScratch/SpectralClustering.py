import pandas as pd
import numpy as np
import random
from scipy import stats
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
random.seed(1)

# -----------------------
# Spectral CLustering on a political blogs graph
# -----------------------

# Import the data
nodes = pd.read_csv('nodes.txt', sep="\t", header=None)
edges = pd.read_csv('edges.txt', sep="\t", header=None)

# Reindex
nodes[0] = nodes[0] - 1
edges[0] = edges[0] - 1
edges[1] = edges[1] - 1

# Keep only the nodes with edges
nodes_with_edges = nodes[nodes[0].isin(edges[0]) | nodes[0].isin(edges[1])]
nodes_with_edges = nodes_with_edges.reset_index(drop=True)

# Remap indexes in 'edges'
node_map = nodes_with_edges[0].reset_index()
edges_new_index=edges
edges_new_index=edges_new_index.replace(node_map.set_index(0)['index'])


# ---------- Compute Normalized Laplacian ----------
def normalized_laplacian(edges_df):
    # Adjacency matrix
    n=len(np.unique(edges_df))
    A = np.zeros((n, n))
    for i, j in zip(edges_df[0], edges_df[1]):
        # Avoid edges from a node to itself
        if i != j:
            A[i, j] = 1
            A[j, i] = 1
    # Degree matrix
    D = np.diag(np.sum(A, axis=1))
    # Identity matrix
    I = np.identity(A.shape[0])
    # Inverse square root of D
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    # Normalized Laplacian
    L_norm = I - np.dot(np.dot(D_inv_sqrt, A), D_inv_sqrt)
    return L_norm


# ---------- Eigendecomposition + select k eigenvectors for clustering ----------
def get_normalized_k_eigenvectors(L_norm, k):
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
    # Since eigh sorts eigenvalues by value, select k eigenvectors from the end
    # These k eigenvectors contain cluster assignment information
    selected_eigenvectors = eigenvectors[:, -k:]
    # Normalize the selected k eigenvectors
    k_eigenvectors_norm = selected_eigenvectors/np.linalg.norm(selected_eigenvectors, axis=1)[:, np.newaxis]
    return k_eigenvectors_norm

A = np.zeros((m, m))
    for i in range(m):
        for j in range(i+1, m):
            d = np.linalg.norm(img_row[i]-img_row[j])
            if d <= epsilon:
                A[i][j] = d
                A[j][i] = d


# ---------- K-means on the k eigenvectors ----------
def run_kmeans(k, k_eigenvectors_norm):
    # Run k means on the k normalized eigenvectors
    # Included random state for replicability
    kmeans = KMeans(n_clusters=k, random_state = 0)
    # Predict labels for each point (0 or 1)
    labels = kmeans.fit_predict(k_eigenvectors_norm)
    return labels


# ---------- Compute & print mismatch rates ----------
def print_mismatch(nodes_with_edges,label_col, k, labels):
    s=''
    s +=f'k={k}:\n'
    total_mismatch_count = 0
    total_points = len(nodes_with_edges)
    mismatch_rates = []
    for c in range(k):
        true_c_labels = np.array(nodes_with_edges[labels==c][label_col])
        c_size = len(true_c_labels)
        mode, mode_count = stats.mode(true_c_labels)
        mismatch_count = c_size-mode_count[0]
        total_mismatch_count += mismatch_count
        mismatch_rate = mismatch_count/c_size
        mismatch_rates.append(mismatch_rate)
        s+=f'Cluster {c}: True label {mode[0]}, cluster size {c_size}, mismatch count {mismatch_count}, mismatch rate {round(mismatch_rate,4)}\n'
    total_rate = total_mismatch_count/total_points
    s+=f'Total mismatch rate:{round(total_rate,4)}\n'
    return s, total_rate



# ---------- Implement the above on given k values ----------
# Compute normalized Laplacian
L_norm = normalized_laplacian(edges_new_index)
# For each k, perform eigendecomposition, run k-means and get the mismatch rate
k_arr = np.array([2,5,10,25])
for k in k_arr:
    # k eigenvectors
    k_eigenvectors_norm = get_normalized_k_eigenvectors(L_norm, k)
    # run kmeans
    labels = run_kmeans(k,k_eigenvectors_norm)
    # print result
    s, _ = print_mismatch(nodes_with_edges,2, k, labels)
    print(s)


# ---------- Repeat the process for k from 2 to 50, visualize total mismatch ----------
k_arr = np.array(range(2,50))
total_rates = []
s_lst=[]
for k in k_arr:
    # k eigenvectors
    k_eigenvectors_norm = get_normalized_k_eigenvectors(L_norm, k)
    # run kmeans
    labels = run_kmeans(k,k_eigenvectors_norm)
    # print result
    s, total_rate = print_mismatch(nodes_with_edges,2, k, labels)
    total_rates.append(total_rate)
    s_lst.append(s)
plt.figure(figsize=(15,15))
plt.plot(k_arr, total_rates, marker='D')
plt.xlabel('k')
plt.ylabel('Total Mismatch Rate')
plt.title('Total Mismatch Rate vs. Number of clusters(k) ')
plt.grid(True)
plt.show()

# ---------- Print out mismatches for the (reasonable) best k ----------
total_rates[19]
print(s_lst[19])
