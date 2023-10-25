import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
np.random.seed(555)


# ---------- Load the data & Preprocess ----------
data = scipy.io.loadmat('data/cs.mat')['img']

# ---------- Plot input data ----------
plt.imshow(data, cmap='gray')
plt.title("Original Image")
plt.show()

# ---------- Create y ----------
img_flat = data.reshape(-1)
# Generate e and A based on distribution info
e = np.random.randn(1300) * np.sqrt(25)
A = np.random.randn(1300,2500)
# Use flattened img array, e and A to get y
y = np.dot(A, img_flat) + e
y = y.reshape(-1)

# ---------- Preform 10-fold CV for LASSO ----------
# Grid search for best alpha
alpha_search = {'alpha':np.arange(0, 1.5, 0.005)}
lasso = Lasso()
cv_lasso = KFold(n_splits=10, random_state=0, shuffle=True)
lasso_grid = GridSearchCV(lasso, alpha_search, scoring='neg_mean_squared_error', cv=cv_lasso, n_jobs=-1, verbose=2)
lasso_grid.fit(A,y)

# Results and parameters after CV
# Add '-' because GridSearch used negative MSEs
lasso_mse = -lasso_grid.cv_results_['mean_test_score']
lasso_mse_std = lasso_grid.cv_results_['std_test_score']
best_alpha_lasso = lasso_grid.best_params_['alpha']
best_mse_lasso = -lasso_grid.best_score_

# ---------- CV Curve LASSO ----------
plt.figure()
plt.plot(alpha_search['alpha'], lasso_mse)

# Visualize MSE StDev
plt.fill_between(alpha_search['alpha'], lasso_mse - lasso_mse_std, lasso_mse + lasso_mse_std, alpha=0.1)

# Indicate the best Alpha and min MSE
plt.scatter(best_alpha_lasso, best_mse_lasso, color='darkred')
plt.text(best_alpha_lasso, best_mse_lasso, f' Best Alpha={best_alpha_lasso:.3f}', verticalalignment='bottom')
plt.xlabel('Alpha Parameter ($\lambda$)')
plt.ylabel('MSE')
plt.title('CV Error Curve for LASSO')
plt.grid(True)
plt.show()

# ---------- Refit LASSO with best lambda ----------
lasso_best_alpha = Lasso(alpha=best_alpha_lasso)
lasso_best_alpha.fit(A, y)

# ---------- Reconstruct image ----------
img_lasso = np.reshape(lasso_best_alpha.coef_, (50, 50))

# ---------- Visualize LASSO reconstructed Image ----------
plt.figure()
plt.imshow(img_lasso, cmap='gray')
plt.title('Reconstructed Image LASSO')
plt.show()

# ---------- Preform 10-fold CV for RIDGE ----------
# Grid search for best alpha
cv_ridge = KFold(n_splits=10, random_state=1, shuffle=True)
alpha_search_ridge = {'alpha':np.arange(0.005,1000,5)}
ridge = Ridge()
ridge_grid = GridSearchCV(ridge, alpha_search_ridge, scoring='neg_mean_squared_error', cv=cv_ridge, n_jobs=-1, verbose=2)
ridge_grid.fit(A,y)

# Results and parameters after CV
# Add '-' because GridSearch used negative MSEs
ridge_mse = -ridge_grid.cv_results_['mean_test_score']
ridge_mse_std = ridge_grid.cv_results_['std_test_score']
best_alpha_ridge = ridge_grid.best_params_['alpha']
best_mse_ridge = -ridge_grid.best_score_

# ---------- CV Curve RIDGE ----------
plt.figure()
plt.plot(alpha_search_ridge['alpha'], ridge_mse)

# Visualize MSE StDev
plt.fill_between(alpha_search_ridge['alpha'], ridge_mse - ridge_mse_std, ridge_mse + ridge_mse_std, alpha=0.1)

# Indicate the best Alpha and min MSE
plt.scatter(best_alpha_ridge, best_mse_ridge, color='darkred')
plt.text(best_alpha_ridge, best_mse_ridge, f' Best Alpha={best_alpha_ridge:.3f}', verticalalignment='bottom')
plt.xlabel('Alpha Parameter ($\lambda$)')
plt.ylabel('MSE')
plt.title('CV Error Curve for Ridge')
plt.grid(True)
plt.show()

# ---------- Refit RIDGE with best lambda ----------
ridge_best_alpha = Ridge(alpha=best_alpha_ridge)
ridge_best_alpha.fit(A, y)

# ---------- Reconstruct image ----------
img_ridge = np.reshape(ridge_best_alpha.coef_, (50, 50))

# ---------- Visualize Ridge reconstructed Image ----------
plt.figure()
plt.imshow(img_ridge, cmap='gray')
plt.title('Reconstructed Image Ridge')
plt.show()