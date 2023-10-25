from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(3680)

sub1_paths = ['data/yalefaces/subject01.glasses.gif',
              'data/yalefaces/subject01.happy.gif',
              'data/yalefaces/subject01.leftlight.gif',
              'data/yalefaces/subject01.noglasses.gif',
              'data/yalefaces/subject01.normal.gif',
              'data/yalefaces/subject01.rightlight.gif',
              'data/yalefaces/subject01.sad.gif',
              'data/yalefaces/subject01.sleepy.gif',
              'data/yalefaces/subject01.surprised.gif',
              'data/yalefaces/subject01.wink.gif',
              'data/yalefaces/subject01-test.gif']

sub2_paths = ['data/yalefaces/subject02.glasses.gif',
              'data/yalefaces/subject02.happy.gif',
              'data/yalefaces/subject02.leftlight.gif',
              'data/yalefaces/subject02.noglasses.gif',
              'data/yalefaces/subject02.normal.gif',
              'data/yalefaces/subject02.rightlight.gif',
              'data/yalefaces/subject02.sad.gif',
              'data/yalefaces/subject02.sleepy.gif',
              'data/yalefaces/subject02.wink.gif',
              'data/yalefaces/subject02-test.gif']


#Original shape of images:
original_shape = Image.open(sub1_paths[0]).size
factor = 4
#----Resize images----
def resize(path_to_img, factor):
    img = Image.open(path_to_img)
    w,h = img.size
    w_new = w // factor
    h_new = h // factor
    resized_img = img.resize((w_new, h_new))
    return resized_img

#----Vectorize images----
def vectorize (resized_img):
    img_as_array = np.array(resized_img)
    vectorized_img = img_as_array.flatten()
    return vectorized_img

# Vectorize subject 1 & 2 test images
sub1_test = vectorize(resize(sub1_paths[-1], factor))
sub2_test = vectorize(resize(sub2_paths[-1], factor))

#----Matrix of vectorized images----
sub1_vectors=[]
sub2_vectors=[]

for img in sub1_paths[:-1]:
    resized_img_s1 = resize(img, factor)
    vector_img_s1 = vectorize(resized_img_s1)
    sub1_vectors.append(vector_img_s1)
sub1_matrix = np.array(sub1_vectors)

for img in sub2_paths[:-1]:
    resized_img_s2 = resize(img, factor)
    vector_img_s2 = vectorize(resized_img_s2)
    sub2_vectors.append(vector_img_s2)
sub2_matrix = np.array(sub2_vectors)

#----PCA----
def pca_extract_components(d_array, k):
    #Perform & fit PCA
    pca = PCA(n_components=k)
    pca.fit(d_array)
    #Get first k PCs
    first_k_pc = pca.components_
    return first_k_pc

# Since we need 6 eigenfaces
k=6
sub1_eigenfaces = pca_extract_components(d_array = sub1_matrix,k = k)
sub2_eigenfaces = pca_extract_components(d_array = sub2_matrix,k = k)

#----Reshape Eigenfaces----
def reshape_back(sub_eigenfaces, original_shape, factor):
    reshaped = []
    w_orig, h_orig = original_shape
    for eigenface in sub_eigenfaces:
        reshaped_img = eigenface.reshape((h_orig//factor, w_orig//factor))
        reshaped.append(reshaped_img)
    return reshaped

sub1_reshaped_result = reshape_back(sub_eigenfaces=sub1_eigenfaces, original_shape=original_shape, factor=factor)
sub2_reshaped_result = reshape_back(sub_eigenfaces=sub2_eigenfaces, original_shape=original_shape, factor=factor)

def plot_eigenfaces(sub1_reshaped, sub2_reshaped):
    cols = len(sub1_reshaped)
    fig, axs = plt.subplots(nrows = 2, ncols = cols, figsize = (18,6))
    for c in range(cols):
        axs[0, c].imshow(sub1_reshaped[c], cmap='gray')
        axs[0, c].set_title(f'PC {c + 1}')
        axs[1, c].imshow(sub2_reshaped[c], cmap='gray')
        axs[1, c].set_title(f'PC {c + 1}')
    fig.text(0.5, 0.9, 'Subject 1', ha='center', fontsize=12)
    fig.text(0.5, 0.45, 'Subject 2', ha='center', fontsize=12)
    return plt.show()
plot_eigenfaces(sub1_reshaped_result, sub2_reshaped_result)

face1 = sub1_eigenfaces[0]
face2 = sub2_eigenfaces[0]

def projection_res(face,test):
    face = face.reshape((-1,1))
    res = test - face@face.T@test
    proj_res = np.linalg.norm(test - face@face.T@test)**2
    return proj_res

s11 = projection_res(face1, sub1_test)
s12 = projection_res(face1, sub2_test)
s21 = projection_res(face2, sub1_test)
s22 = projection_res(face2, sub2_test)


projection_residuals_df = pd.DataFrame({'Subject': [1,1,2,2],
                                        'Test': [1,2,1,2],
                                        'Projection residual': [s11, s12, s21, s22]})

print(projection_residuals_df)
