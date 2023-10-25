import pandas as pd
import numpy as np
import scipy.sparse.linalg as ll
import matplotlib.pyplot as plt
np.random.seed(520)

# -----------------------
# PCA From Scratch 
# -----------------------

# Import the data
data = pd.read_csv("data/food-consumption.csv")
# Extract country labels
country_labels = data['Country'].values
# Extract food labels
food_labels = data.columns[1:].values
# Features (columns)
food = data.iloc[:, 1:].values

# -----------------------
# Define the functions
# -----------------------

#----Standardize the data----
def standardize(x):
    # ------------------------------------------------------
    # Standardizes the input data by scaling & centering,
    # Assumes (features = columns).
    # INPUT: Data set as a np.array
    # OUTPUT: Standardized data set (np.array)
    # ------------------------------------------------------
    mean_x = np.mean(x, axis=0)
    std_x = np.std(x, axis=0)
    x_standardized = (x - mean_x) / std_x
    return x_standardized

#----Compute the covariance matrix----
def covariance_matrix(x_standardized):
    # ------------------------------------------------------
    # Computes the covariance matrix.
    # INPUT: Standardized data set (np.array)
    # OUTPUT: Covariance matrix (m*m, m - # of features)
    # ------------------------------------------------------
    covariance_mat = (x_standardized.T @ x_standardized) / (x_standardized.shape[0])
    return covariance_mat

#----Eigendecomposition: k directions----
def k_directions(covariance_mat, k):
    # ------------------------------------------------------
    # Performs eigendecomposition and returns k eigenvectors
    # corresponding to the k largest eigenvalues.
    # INPUT: Covariance matrix
    # OUTPUT: S: the k largest eigenvalues of shape (k,)
    #         W: the k corresponding eigenvectors, of shape
    #            (m,k), m - # of features
    # ------------------------------------------------------
    S, W = ll.eigs(covariance_mat, k = k, which='LM')
    S = S.real
    W = W.real
    return S, W

#----Compute the first k principal components by projecting the data----
def project_PCs(x_standardized, S, W, k):
    # ------------------------------------------------------
    # Computes the first k principal components capturing
    # the most variability in the data.
    # INPUT: standardized data, k eigenvalues & eigenvectors, k
    # OUTPUT: PCs: a list of k first principal components
    # ------------------------------------------------------
    PCs = []
    for i in range(k):
        pc_i = np.dot(x_standardized, W[:, i]) / np.sqrt(S[i])
        PCs.append(pc_i)
    return PCs

#----Perform PCa----
def pca(x, k):
    x_standardized = standardize(x = x)
    covariance_mat = covariance_matrix(x_standardized = x_standardized)
    S, W = k_directions(covariance_mat = covariance_mat, k = k)
    pc = project_PCs(x_standardized = x_standardized, S = S, W = W, k = k)
    return pc

#----Visualize two principal components on a scatter plot----
def visualize_two_pcs(pc1, pc2, labels):
    # ------------------------------------------------------
    # Creates a scatter plot of the two principal components.
    # INPUT: principal components (pc1, pc2), data labels.
    # OUTPUT: PCs: a list of k first principal components
    # ------------------------------------------------------
    plt.scatter(pc1, pc2)
    plt.grid()
    for i, f in enumerate(labels):
        plt.annotate(f, (pc1[i], pc2[i]))
    plt.title('The First Two Principal Components')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    #plt.savefig('output_plot.png', transparent=False)
    return plt.show()


# Run the functions on the data set where features = columns
# First two principal components:
k = 2
pc1, pc2 = pca(x = food, k = k)

# Visualize the result
# Use country names as labels:
visualize_two_pcs(pc1 = pc1, pc2 = pc2, labels = country_labels)

# Features (rows)
transposed = np.transpose(food)

# --------------------------
# 2.2.1 Implementation
# --------------------------

# Implement PCA. Food item names are the new labels
pc1_1, pc2_1 = pca(x = transposed, k = k)
# Visualize the result:
# Use country names as labels
visualize_two_pcs(pc1 = pc1_1, pc2 = pc2_1, labels = food_labels)
