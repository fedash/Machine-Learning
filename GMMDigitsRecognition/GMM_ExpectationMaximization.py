import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import scipy.io
from scipy.sparse.linalg import eigs
import matplotlib.ticker as ticker
from scipy import stats
from sklearn.cluster import KMeans

np.random.seed(555)


# ---------- Load the data & Preprocess ----------
data = scipy.io.loadmat('data/data.mat')['data']
label = scipy.io.loadmat('data/label.mat')['trueLabel']
m, n = data.shape


# ---------- Implement PCA with 4 directions ----------
C = np.matmul(data, data.T) / m
d = 4
vals, V = eigs(C, k=d, which='LM')
vals = vals.real
V = V.real
proj_data = np.dot(data.T, V)


# ---------- Implement EM for GMM ----------
# 1. Initialize parameters
# Number of Gaussian components
K = 2

# Initialize the prior
pi = np.random.rand(K)
pi = pi/np.sum(pi)

# Initialize the mean
mu = np.random.randn(K, d)

# Initialize the covariance
sigma = []
for r in range(K):
    S = np.random.randn(d, d)
    sigma.append(S @ S.T + np.eye(d))

# Store log-likelihoods of iterations
loglikelihoods = []

# 2. Iterate E & M
# Initiate log-likelihood
ll_0 = 0
for i in range(100):

    # Expectation
    # Initiate responsibilities
    tau = np.zeros((K, n))
    for k in range(K):
        # Compute responsibility for each Gaussian
        tau[k, :] = pi[k] * mvn.pdf(proj_data, mu[k], sigma[k])
    tau = tau / tau.sum(0)

    # Maximization
    # Prior & Mean are now based on E-step results
    pi = np.mean(tau, axis=1) #for each row
    mu = np.dot(tau, proj_data) / tau.sum(1)[:, None] #by rows
    for k in range(K):
        x_mu = proj_data - mu[k, :]
        sigma[k] = x_mu.T @ np.diag(tau[k, :]) @ x_mu / tau[k, :].sum()

    # Compute log-likelihood
    ll_1 = 0
    for k in range(K):
        ll_1 += pi[k] * mvn.pdf(proj_data, mu[k], sigma[k])
    ll_1 = np.log(ll_1).sum()

    # Converged?
    if np.abs(ll_1 - ll_0) < 1e-3:
        break
    ll_0 = ll_1
    loglikelihoods.append(ll_1)
print(f"Converged after iteration {i}. Log-likelihood: {ll_1:.2f}")


# ---------- Plot log-likelihood VS iteration ----------
plt.plot(loglikelihoods, marker='o', color='steelblue')
plt.grid(True)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.title("Log-likelihood VS Iterations")
plt.xlabel("Iteration")
plt.ylabel("Log-likelihood")
plt.show()


# ---------- Report the result & Visualize ----------
for k in range(K):
    # Report component weights
    print(f"\nComponent {k + 1} has weight {pi[k]:.4f}")

    # Mean of each Gaussian
    mean_old = np.dot(mu[k], V.T)
    mean_comp = mean_old.reshape((28, 28), order='F')
    plt.figure()
    plt.imshow(mean_comp, cmap='gray')
    plt.title(f"Mean of Gaussian Component {k + 1}")
    plt.show()

    # Covariance matrix of each Gaussian
    cov_map = np.dot(V, np.dot(sigma[k], V.T))
    plt.imshow(cov_map, cmap='binary')
    plt.colorbar()
    plt.title(f"Covariance of Gaussian Component {k + 1}")
    plt.show()

    # Heatmap version
    covm = sigma[k]
    plt.imshow(covm, cmap='binary')
    plt.colorbar()
    plt.title(f"Covariance matrix of Gaussian Component {k + 1}")
    plt.show()


# ---------- Misclassification rates GMM ----------
# Assign each sample to the component with larger responsibility (0 or 1)
predicted_labels = np.argmax(tau, axis=0)

# Map predicted label values to true ones (0,1 -> 2,6)
map_labels = {}
for pred_label in np.unique(predicted_labels):
    original_labels = label.T[predicted_labels == pred_label]
    map_labels[pred_label] = stats.mode(original_labels).mode[0][0]

# Predicted labels in true values
predicted_labels_repl = np.array([map_labels[pred_label] for pred_label in predicted_labels])

# Misclassification for '2' and '6'
mc_2_gmm = np.mean(predicted_labels_repl[label.flatten()==2] != 2)
print(f"\nMisclassification rate for digit '2': {mc_2_gmm:.5f} (GMM)")
mc_6_gmm = np.mean(predicted_labels_repl[label.flatten()==6] != 6)
print(f"\nMisclassification rate for digit '6': {mc_6_gmm:.5f} (GMM)")

# Misclassification for both
mc_total_gmm = np.mean(label!=predicted_labels_repl)
print(f"\nTotal misclassification rate: {mc_total_gmm:.5f} (GMM)")


# ---------- Misclassification rates KMeans ----------
kmeans = KMeans(n_clusters=K, random_state=0).fit(proj_data)
kmeans_pred_labels = kmeans.labels_

# Map predicted label values to true ones (0,1 -> 2,6)
map_labels_kmeans = {}
for pred_label in np.unique(kmeans_pred_labels):
    original_labels = label.T[kmeans_pred_labels == pred_label]
    map_labels_kmeans[pred_label] = stats.mode(original_labels).mode[0][0]

# Predicted labels in true values
kmeans_pred_labels_repl = np.array([map_labels_kmeans[pred_label] for pred_label in kmeans_pred_labels])

# Misclassification for '2' and '6'
mc_2_kmeans = np.mean(kmeans_pred_labels_repl[label.flatten()==2] != 2)
print(f"\nMisclassification rate for digit '2': {mc_2_kmeans:.5f} (KMeans)")

mc_6_kmeans = np.mean(kmeans_pred_labels_repl[label.flatten()==6] != 6)
print(f"\nMisclassification rate for digit '6': {mc_6_kmeans:.5f} (KMeans)")

# Misclassification for both
mc_total_kmeans = np.mean(label.flatten()!=kmeans_pred_labels_repl)
print(f"\nTotal misclassification rate: {mc_total_kmeans:.5f} (KMeans)")
