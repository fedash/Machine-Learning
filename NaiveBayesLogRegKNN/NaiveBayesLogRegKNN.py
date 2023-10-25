import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, log_loss
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
np.random.seed(555)

# ---------- Load the data & Preprocess ----------
data = pd.read_csv('data/marriage.csv', header=None)
# Store features
X = data.iloc[:, :-1]
# Labels in the last column
y = data.iloc[:, -1]

# ---------- Split into train and test sets ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- Scale (since we will do KNN too) ----------
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# ---------- Naive Bayes ----------
gnb = GaussianNB(var_smoothing=1e-3)
gnb.fit(scaled_X_train, y_train)
accuracy_nb_test = gnb.score(scaled_X_test, y_test)
print(f'Naive Bayes Testing Accuracy: {accuracy_nb_test:.5f}')

# ---------- Logistic Regression ----------
logreg = LogisticRegression( )
logreg.fit(scaled_X_train, y_train)
accuracy_logreg_test = logreg.score(scaled_X_test, y_test)
print(f'Logistic Regression Testing Accuracy: {accuracy_logreg_test:.5f}')

# ---------- KNN ----------
# Grid search to get best n_neigbors
k = list(range(1,55))
knn = KNeighborsClassifier()
kgrid = GridSearchCV(knn, dict(n_neighbors=k), cv=10, scoring='accuracy', return_train_score=False)
kgrid.fit(scaled_X_train, y_train)
best_n = kgrid.best_params_['n_neighbors']
print(f"Best number of neighbors for KNN: {best_n}")
#Train with best k
knn_best = KNeighborsClassifier(n_neighbors=best_n)
knn_best.fit(scaled_X_train, y_train)
accuracy_knn_test = knn_best.score(scaled_X_test, y_test)
print(f'KNN Testing Accuracy: {accuracy_knn_test:.5f}')

# ---------- Compare performance ----------
#Predict
y_pred_nb = gnb.predict(scaled_X_test)
y_pred_logreg = logreg.predict(scaled_X_test)
y_pred_knn = knn_best.predict(scaled_X_test)
# Report Classification scores
print("Naive Bayes Report:\n", classification_report(y_test, y_pred_nb))
print("Logistic Regression Report:\n", classification_report(y_test, y_pred_logreg))
print("KNN Report:\n", classification_report(y_test, y_pred_knn))

# ---------- PCA ----------
pca = PCA(n_components=2)
pca_X_train = pca.fit_transform(scaled_X_train)
pca_X_test = pca.transform(scaled_X_test)

# ---------- Naive Bayes on PCA ----------
gnb_pca = GaussianNB(var_smoothing=1e-3)
gnb_pca.fit(pca_X_train, y_train)
accuracy_nb_pca = gnb_pca.score(pca_X_test, y_test)
print(f'Naive Bayes Testing Accuracy (after PCA): {accuracy_nb_pca:.5f}')

# ---------- Logistic Regression on PCA ----------
logreg_pca = LogisticRegression()
logreg_pca.fit(pca_X_train, y_train)
accuracy_logreg_pca = logreg_pca.score(pca_X_test, y_test)
print(f'Logistic regression Testing Accuracy (after PCA): {accuracy_logreg_pca:.5f}')

# ---------- KNN on PCA ----------
kgrid_pca = GridSearchCV(knn, dict(n_neighbors=k), cv=10, scoring='accuracy', return_train_score=False)
kgrid_pca.fit(pca_X_train, y_train)
best_n_pca = kgrid_pca.best_params_['n_neighbors']
print(f"Best number of neighbors for KNN (after PCA): {best_n_pca}")
knn_best_pca = KNeighborsClassifier(n_neighbors=best_n_pca)
knn_best_pca.fit(pca_X_train, y_train)
accuracy_knn_pca = knn_best_pca.score(pca_X_test, y_test)
print(f'KNN Testing Accuracy (after PCA): {accuracy_knn_pca:.5f}')

# ---------- Plot boundaries ----------
def plot_boundary(X, y, clf, title):
    plot_decision_regions(X, y.astype(int).values, clf=clf, legend=2)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.show()

plot_boundary(pca_X_test, y_test, gnb_pca, 'Naive Bayes')
plot_boundary(pca_X_test, y_test, logreg_pca, 'Logistic Regression')
plot_boundary(pca_X_test, y_test, knn_best_pca, 'KNN')

# ---------- Compare performance ----------
#Predict
y_pred_nb_pca = gnb_pca.predict(pca_X_test)
y_pred_logreg_pca = logreg_pca.predict(pca_X_test)
y_pred_knn_pca = knn_best_pca.predict(pca_X_test)
# Report Classification scores
print("Naive Bayes Report:\n", classification_report(y_test, y_pred_nb_pca))
print("Logistic Regression Report:\n", classification_report(y_test, y_pred_logreg_pca))
print("KNN Report:\n", classification_report(y_test, y_pred_knn_pca))