from scipy.sparse.csgraph import shortest_path
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import random
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.spatial import distance
import networkx as nx

# Import the data
mat = scipy.io.loadmat('data/isomap.mat')
data = mat['images']
img_row = np.transpose(data)
m,n = img_row.shape
k=2

# -----------------------
# QUESTION 3.A
# -----------------------

# -----------------------
# Define the functions
# -----------------------


#----Estimate the best epsilon range----
def pairwise_dist_hist(data):
    pairwise_distances = distance.pdist(data)
    m,_ = data.shape
    bbins = int(m/4)
    plt.hist(pairwise_distances, bins=bbins, density=True, color='steelblue', zorder=3)
    plt.grid(zorder=0)
    median = np.median(pairwise_distances)
    mean = np.mean(pairwise_distances)
    std_dev = np.std(pairwise_distances)
    plt.axvline(median, color='darkred', linestyle='dashed', linewidth=1.5,
                label='Median: {:.0f}'.format(median), zorder=5)
    plt.axvline(mean, color='darkorange', linestyle='dashed', linewidth=1.5,
                label='Mean: {:.0f}'.format(mean), zorder=5)
    plt.axvline(mean - std_dev, color='yellow', linestyle='dashed', linewidth=1.5,
                label='Mean - 1sd: {:.0f}'.format(mean - std_dev), zorder=5)
    plt.axvline(mean + std_dev, color='green', linestyle='dashed', linewidth=1.5,
                label='Mean + 1sd: {:.0f}'.format(mean + std_dev), zorder=5)
    plt.axvline(mean - 2*std_dev, color='pink', linestyle='dashed', linewidth=1.5,
                label='Mean - 2sd: {:.0f}'.format(mean - 2*std_dev), zorder=5)
    plt.axvline(mean + 2*std_dev, color='purple', linestyle='dashed', linewidth=1.5,
                label='Mean + 2sd: {:.0f}'.format(mean + 2*std_dev), zorder=5)
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Histogram of Pairwise Distances')
    plt.legend()
    #plt.savefig("pdist_hist.png", format="png", transparent=False)
    return plt.show()

#----Compute the WNN graph, visualize as a matrix----
def wnn_graph(data, epsilon, max_dist=None):
    m,n = data.shape
    A = np.zeros((m,m))
    for i in range(m):
        for j in range(i + 1, m):
            l2_dist = np.linalg.norm(data[i] - data[j])
            if l2_dist <= epsilon:
                A[i][j] = l2_dist
                A[j][i] = l2_dist
    if max_dist is None:
        plt.imshow(A, cmap='viridis', vmin=0, vmax=epsilon)
    else:
        plt.imshow(A, cmap='viridis', vmin=0, vmax=max_dist)
    plt.colorbar()
    plt.title('Weighted Nearest Neighbor Graph, ε={e}'.format(e=epsilon))
    #plt.savefig("WNNG_e={e}.png".format(e=epsilon), format="png", transparent=False)
    plt.show()
    return A

#----Visualize the WNN graph with image labels----
def visualize_wnng(A, data, epsilon):
    random.seed(3680)
    # 1. Create networkx Graph based on the WNNG:
    G = nx.from_numpy_matrix(A)
    # 2. From all nodes take a random sample for labeling:
    nodes = list(G.nodes())
    selected_nodes = random.sample(nodes, 150)
    # 3. Reshape images in the data set to display them as labels, store in a list
    reshaped_images = [np.rot90(img.reshape((64, 64)), k=3) for img in data]  # reshape & rotate
    # 4. Create label images which to display on the graph plot, in reduced size and gray color scale
    img_labels = [OffsetImage(reshaped_images[n], zoom=0.5, cmap='gray') for n in selected_nodes]
    # 5. Initiate the plot space
    fig, ax = plt.subplots(figsize=(25, 25))
    # 5. Draw the graph using networkX
    pos = nx.spring_layout(G, seed=111) # Use the standard layout with seed for reproducible results
    nx.draw(G, pos=pos, with_labels=False, node_size=80, node_color='lightblue', linewidths=0.2, edge_color='gray', width=0.1)
    # 6. Redraw the selected nodes in red to distinguish them from the restDraw the selected nodes in red
    nx.draw_networkx_nodes(G, pos, nodelist=selected_nodes, node_color='red', node_size=80)
    # 7. Label the selected nodes with corresponding original images
    for i in range(150):
        ab = AnnotationBbox(img_labels[i], pos[selected_nodes[i]], xybox=(50., 50.),boxcoords="offset points",
                            pad=0.3, arrowprops=dict(arrowstyle="->,head_width=0.1,head_length=0.1"))
        ax.add_artist(ab)
    plt.title('WNNG visualization for ε={e}'.format(e=epsilon))
    #plt.savefig("WNNG_face_labels_e{e}.png".format(e=epsilon), format="png", transparent=False,dpi=500)
    return plt.show()


pairwise_dist_hist(data = img_row)
#epsilons = [11,12,13,14] #Compare results for epsilon range
#for e in epsilons:
#    A = wnn_graph(data = img_row, epsilon = e, max_dist = 30)
#    visualize_wnng(A = A, data = img_row, epsilon = e)

#Set e=12 (seems to yield the best result)
e = 12
A = wnn_graph(data = img_row, epsilon = e, max_dist = 30)
visualize_wnng(A = A, data = img_row, epsilon = e)

# -----------------------
# QUESTION 3.B
# -----------------------
#----Implement ISOMAP----
def run_isomap(A, k):
    # 1. Pairwise shortest dist matrix D 698*698
    D = shortest_path(A)
    # 2. Centering matrix: H = I - 1/m * 11^T, C = -1/2*H*(D^2)*H
    m = D.shape[0]
    I = np.identity(m)
    ones_mat = np.ones(m)
    H = I - (1/m) * np.outer(ones_mat, ones_mat.T)
    C = -0.5 * H @ np.square(D) @ H
    # 3. Leading eigenvectors and eigenvalues
    S, W = np.linalg.eigh(C)
    sortedS = S.argsort()[::-1][:k]
    S = S[sortedS]
    W = W[:, sortedS]
    # 4. Projected k-d dataset
    Z = W @ np.diag(np.sqrt(S))
    return Z

#----Visualize with image labels----
def visualize_isomap_or_pca(Z, data, epsilon=None):
    #Same logic as visualize_wnng, but a scatterplot
    random.seed(123)
    reshaped_images = [np.rot90(img.reshape((64, 64)), k=3) for img in data]
    selected_nodes = random.sample(list(range(len(data))), 90)
    img_labels = [OffsetImage(reshaped_images[n], zoom=0.5, cmap='gray') for n in selected_nodes]
    # 4. Initiate the plot space
    fig, ax = plt.subplots(figsize=(20, 16))#
    # 5. Plot a scatter plot using Matplotlib
    ax.scatter(Z[:, 0], Z[:, 1], s=80, color='steelblue', linewidths=0.2)
    # 6. Recolor the selected nodes in red to distinguish them from the rest
    ax.scatter(Z[selected_nodes, 0], Z[selected_nodes, 1], s=80, color='darkred')
    # 7. Label the selected nodes with corresponding original images
    for i, node in enumerate(selected_nodes):
        ab = AnnotationBbox(img_labels[i], (Z[node, 0], Z[node, 1]), xybox=(50., 50.), boxcoords="offset points",
                            pad=0.3, arrowprops=dict(arrowstyle="->,head_width=0.1,head_length=0.1"))
        ax.add_artist(ab)
    if epsilon != None:
        plt.title('ISOMAP Results, ε={e}'.format(e=epsilon))
        #plt.savefig("ISOMAP_e{e}.png".format(e=epsilon), format="png", transparent=False, dpi=500)
    else:
        plt.title('PCA Results')
        #plt.savefig("PCA.png".format(e=epsilon), format="png", transparent=False, dpi=500)
    return plt.show()


Z = run_isomap(A = A, k = k)
visualize_isomap_or_pca(Z = Z, data = img_row, epsilon = e)


# -----------------------
# QUESTION 3.C
# -----------------------

#----PCA----
def run_pca(data,k):
    covariance_mat = (data.T @ data) / (data.shape[0])
    #S, W = ll.eigs(covariance_mat, k = k, which='LM')
    #S = S.real
    #W = W.real
    S, W = np.linalg.eigh(covariance_mat)
    sortedS = S.argsort()[::-1][:k]
    S = S[sortedS]
    W = W[:, sortedS]
    PCs = []
    for i in range(k):
        pc_i = np.dot(data, W[:, i]) / np.sqrt(S[i])
        PCs.append(pc_i)
    return PCs


pc_faces1, pc_faces2 = run_pca(data = img_row, k = k)
pc_faces = np.column_stack((pc_faces1, pc_faces2))
visualize_isomap_or_pca(Z=pc_faces, data=img_row)