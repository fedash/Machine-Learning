# Import the packages
import random
import numpy as np
from sklearn import metrics
from skimage.io import imread, imshow, imsave
from matplotlib import pyplot as plt
import time
random.seed(11)

# -----------------------
# QUESTIONS 3.1 & 3.2
# -----------------------


# ---------- Transform the input image ----------
def preprocess_image(path):
    """
    Preprocesses the initial input image by flattening its 3D np array
    representation to a 2D array which can be further used for k-means.
    ---------------------
            Input
    ---------------------
    path: Path to image
    ---------------------
            Output
    ---------------------
    pixels: Image represented by a 2D np array, where each row is a data
            point and each column corresponds to R, G, B color components
    orig_shape: Shape of the original input image
    """
    # 1. Read the input image (represented by a 3D array)
    img = imread(path)

    # 2. Get the shape of the original image
    orig_shape = img.shape

    # 3. Reshape the image to 2D to use for kmeans
    pixels = img.reshape((orig_shape[0] * orig_shape[1]), orig_shape[2])

    return pixels, orig_shape, img


# ---------- K-means implementation ----------
def kmeans(k, pixels):
    """
    Implements K-means with random centroid initialization.
    ---------------------
            Input
    ---------------------
    k: The number of desired clusters
    pixels: The input image where each row is a data point and
            each column corresponds to R, G, B components
    ---------------------
            Output
    ---------------------
    assigned_c:  Column vector with cluster assignment of each
                 data point in pixels ('class')
    new_centroids: Location of k centroids represented with a
                 (K*3) matrix ('centroid')
    iters: The number of iterations it took k-means to converge
    empty: The number of empty clusters detected
    """

    # 1. Initialize centroids
    initial = random.sample(range(pixels.shape[0]), k)
    centroids = pixels[initial]

    # 2. Assign & recalculate
    max_iters = 500
    iters = 1
    empty = 0

    while (iters <= max_iters):

        # 2.1 Calculate L2 distance from each datapoint to each centroid
        l2 = metrics.pairwise_distances(pixels, centroids, metric='euclidean')

        # 2.2 Assign each point to the closest cluster
        closest = np.argmin(l2, axis=1)

        # 2.3 Calculate new centroids
        new_centroids = []  # as a list to change in case of empty clusters
        for c in range(k):

            # 2.3.1 Find the points that belong to cluster c
            points_in = pixels[closest == c]

            # 2.3.2 If c is not empty, set the new centroid as the mean of these points
            if len(points_in) != 0:
                new_centroids.append(np.mean(points_in, axis=0))

            # 2.3.3 If c is empty, keep track of it for future output
            else:
                empty += 1

        # 2.4 Check if the algorithm has converged
        if len(new_centroids) == len(centroids) and np.allclose(new_centroids, centroids):
            break

        # 2.5 For next iteration, k is the number of non-empty clusters
        k = len(new_centroids)
        new_centroids = np.array(new_centroids)

        # 2.6 If no convergence, update centroids and keep iterating
        centroids = new_centroids
        iters += 1

    # 3. Assign each data point to one of the final updated nearest centroids
    l2_fin = metrics.pairwise_distances(pixels, new_centroids, metric='euclidean')
    assigned_c = np.argmin(l2_fin, axis=1)

    return assigned_c, new_centroids, iters, empty


# ---------- Transform clustering results back to the original image's shape ----------
def output_image(k, pixels, classes, centroid, shape):
    """
    Uses centroid location and cluster assignment from k-means
    results to transform the image back to the orginial shape.
    ---------------------
            Input
    ---------------------
    k: The number of cluster used
    pixels: The input image where each row is a data point and
            each column corresponds to R, G, B components
    classes: Column vector with cluster assignment of each
             data point in pixels ('class')
    centroid: Location of k centroids represented with a
              (K*3) matrix ('centroid')
    ---------------------
            Output
    ---------------------
    output_image: Image transformed to original shape
    """

    # 1. Create the output image using kmeans clustering results
    new_image = np.zeros_like(pixels)
    for c in range(k):
        # 1.1 Find the points within cluster c
        c_pixels = np.where(classes == c)

        # 1.2 Replace those pixels with the centroid for cluster c
        new_image[c_pixels] = centroid[c]  # 2D np array, shape (pixels,3)

    # 2. Reshape new_image to match the original image shape
    output_image = new_image.reshape(shape)  # 3D np array, shape (n,m,3)

    return output_image


# ---------- Implementing the above functions on three input images ----------

k_arr = np.array([2, 3, 4, 5, 6])
for img_path in np.array(['football.bmp', 'hestain.bmp', 'lake.bmp']):

    # Figure to display results
    disp = plt.figure(figsize=(20, 20))
    rw = 3
    cl = 2
    disp.add_subplot(rw, cl, 1)

    # 1. Preprocess
    img_pixels, img_shape, img_original = preprocess_image(path=img_path)

    # Display original image
    plt.imshow(img_original)
    plt.axis('off')
    plt.title("Original")

    for curr_k in k_arr:
        # 1. Start timer
        start_time = time.time()

        # 2. Run K-means
        cluster_result, centroid_result, num_iter, empty_c = kmeans(k=curr_k,
                                                                    pixels=img_pixels)

        # 3. Output image
        out_img = output_image(k=curr_k,
                               pixels=img_pixels,
                               classes=cluster_result,
                               centroid=centroid_result,
                               shape=img_shape)

        # Calculate the elapsed time
        end_time = time.time()
        time_passed = round(end_time - start_time, 3)

        # Display image, #. of iterations and elapsed time
        disp.add_subplot(rw, cl, curr_k)
        plt.imshow(out_img)
        plt.axis('off')
        plt.title(f"k={curr_k}, empty: {empty_c}, iterations: {num_iter}, time: {time_passed} seconds.")

        # save the resulting image for current k
        # imsave(f'k_{curr_k}_{img_path}.bmp', out_img)
        # save iterations & time results for different k
        # plt.savefig(f'{img_path}.png', facecolor='white', transparent=False, bbox_inches='tight')
    plt.show()



# -----------------------
# QUESTION 3.3
# -----------------------



# ---------- Calculating WCSS ----------
def calculate_wcss(pixels, cluster_result, centroid_result):
    wcss = 0
    for c in range(len(centroid_result)):
        points_in_cluster = img_pixels[cluster_result == c]
        centroid = centroid_result[c]
        d = points_in_cluster - centroid
        wcss += np.sum(np.linalg.norm(d/255, axis=1) ** 2) #standardized

    return wcss



# ---------- Get WCSS for different k to find the best one ----------
#*NOTE* For the report I used k values from 2 to 30 for each image,
#       which runs for ~12 mins due to high k and many iterations.

k_arr = np.arange(2, 7)
wcss_results = {'football.bmp': [], 'hestain.bmp': [], 'lake.bmp': []}
for img_path in np.array(['football.bmp', 'hestain.bmp', 'lake.bmp']):
    # 1. Preprocess
    img_pixels, img_shape, img_original = preprocess_image(path = img_path)
    # 2. Run kmeans
    for curr_k in k_arr:
        cluster_result, centroid_result, _, _ = kmeans(k = curr_k, pixels = img_pixels)
        wcss = calculate_wcss(img_pixels, cluster_result, centroid_result)
        wcss_results[img_path].append(wcss)

# 3. Plot WCSS results
plt.figure(figsize=(15, 10))
for img_path, wcss_val in wcss_results.items():
    plt.plot(k_arr, wcss_val, label=img_path)
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.title('Within-Cluster Sum of Squares for different k')
plt.legend()
plt.grid(True)
#plt.savefig(f'WCSS.png', facecolor='white', transparent=False)
plt.show()