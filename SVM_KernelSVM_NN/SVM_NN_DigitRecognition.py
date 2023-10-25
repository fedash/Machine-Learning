import numpy as np
import scipy.io
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(555)


# ---------- Load the data & Preprocess ----------
data = scipy.io.loadmat('data/mnist_10digits.mat')

# Standardize
xtrain = data['xtrain'] / 255
ytrain = data['ytrain']
ytrain = ytrain.reshape(-1)
xtest = data['xtest'] / 255
ytest = data['ytest']
ytest = ytest.reshape(-1)

# Downsample for KNN and SVM
xtrain_res, ytrain_res = resample(xtrain, ytrain, n_samples=5000, random_state=555)

# ---------- KNN ----------

# Grid search to get best n_neigbors - commented out to improve speed
# k = list(range(1,11))
# knn = KNeighborsClassifier()
# kgrid = GridSearchCV(knn, dict(n_neighbors=k, metric=['euclidean', 'manhattan', 'chebyshev', 'minkowski']), cv=5, scoring='accuracy', return_train_score=False, verbose=2)
# kgrid.fit(xtrain_res, ytrain_res)
# best_n = kgrid.best_params_['n_neighbors']
# best_metric = kgrid.best_params_['metric']

# print(f"Best number of neighbors for KNN: {best_n}, best distance metric: {best_metric}")

# Train with best k and distance and fit to full dataset
best_n = 3
best_metric = 'euclidean'
knn_best = KNeighborsClassifier(n_neighbors=best_n, metric=best_metric)
# For the report - fit on full data set:
# knn_best.fit(xtrain, ytrain)
# To improve speed not, used resampled data:
knn_best.fit(xtrain_res, ytrain_res)
y_pred_knn = knn_best.predict(xtest)
def plot_cm(true, pred, title):
    cm = confusion_matrix(true, pred)
    sns.heatmap(cm,annot=True, fmt=".0f", cmap = 'BuPu');
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.show()

# Report performance
print(f"\nKNN Performance (with best n_neighbors={best_n}, , best distance metric: {best_metric}): ")
plot_cm(ytest, y_pred_knn, title='Confusion Matrix for KNN')
print(classification_report(ytest, y_pred_knn))

# ---------- Logistic regression ----------

# Grid search to find best parameters - commented out to improve speed
# logreg = LogisticRegression( )
# logreg_search = [{'penalty' : ['l1', 'l2', 'elasticnet'],
#                   'C' : np.arange(0, 1.1, 0.1),
#                   'solver' : ['lbfgs','newton-cg','liblinear','sag','saga']}]
# cvlogreg = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
# logreg_grid = GridSearchCV(estimator = logreg, param_grid = logreg_search, cv = cvlogreg, n_jobs = -1, verbose = 2)
# logreg_grid.fit(xtrain_res, ytrain_res)
# logreg_grid.best_params_
# print(f"Best LogReg Parameters: {logreg_grid.best_params_}")

#Train with the best parameters and fit to full dataset
best_penalty='l2'
best_solver='lbfgs'
best_c=0.1
best_logreg = LogisticRegression(penalty=best_penalty, solver=best_solver, C=best_c, max_iter=500)
best_logreg.fit(xtrain, ytrain)
y_pred_logreg = best_logreg.predict(xtest)

# Report Performance
print(f"\nLogistic Regression Performance (with best parameters penalty={best_penalty}, solver={best_solver}, C={best_c}): ")
plot_cm(ytest, y_pred_logreg, title='Confusion Matrix for Logistic Regression')
print(classification_report(ytest, y_pred_logreg))

# ---------- SVM ----------

# Grid search to find best parameters - commented out to improve speed
# svmsearch = {'C': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 3, 5, 7, 10, 15]}
# svm = SVC(kernel='linear')
# cvsvm = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
# svm_grid = GridSearchCV(svm, svmsearch, cv=cvsvm, verbose=2, n_jobs=-1)
# svm_grid.fit(xtrain_res, ytrain_res)
# print(f"Best SVM C: {svm_grid.best_params_}")

# Train with the best parameters and fit to resmapled dataset (full takes too much time)
best_c = 0.1
best_svm = SVC(C=best_c, kernel='linear', random_state=17)
# For the report - fit on full data set:
# best_svm.fit(xtrain, ytrain)
# To improve speed not, used resampled data:
best_svm.fit(xtrain_res, ytrain_res)
y_pred_svm = best_svm.predict(xtest)

# Report Performance
print(f"\nLinear SVM Performance (with best C={best_c}): ")
plot_cm(ytest, y_pred_svm, title='Confusion Matrix for SVM')
print(classification_report(ytest, y_pred_svm))

# ---------- Kernel SVM ----------

#Grid search to find best parameters
# ksvm_search = {'C': [0.1, 1, 5, 10, 15],
#               'kernel': ['linear', 'poly', 'rbf'],
#               'gamma': ['scale', 'auto']}  #used by rbf and poly
# ksvm = SVC()
# cvksvm = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
# ksvm_grid = GridSearchCV(ksvm, ksvm_search, cv=cvksvm, verbose=2, n_jobs=-1)
# ksvm_grid.fit(xtrain_res, ytrain_res)
# print(f"Best SVM Parameters: {ksvm_grid.best_params_}")

# Train with the best parameters and fit to resmapled dataset
best_c = 5
best_gamma = 'scale'
best_kernel = 'rbf'
# Fit on full data set as per instructions
ksvm_best = SVC(C=best_c, kernel=best_kernel, gamma=best_gamma)
ksvm_best.fit(xtrain, ytrain)

# Report performance
y_pred_ksvm = ksvm_best.predict(xtest)
print("\nKernel SVM Performance: ")
plot_cm(ytest, y_pred_ksvm, title='Confusion Matrix for Kernel SVM')
print(classification_report(ytest, y_pred_ksvm))

# ---------- NN ----------

#Grid search to find best parameters
# mlp = MLPClassifier(hidden_layer_sizes=(20,10), random_state=17)
# mlp_search = {'activation': ['tanh', 'relu'],
#               'solver': ['sgd', 'adam'],
#               'alpha': [0.0001, 0.01, 0.1, 1, 5],
#               'learning_rate': ['constant','adaptive']}
# cvmlp = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
# mlp_grid = GridSearchCV(mlp, mlp_search, cv=cvmlp, verbose=2, n_jobs=-1)
# mlp_grid.fit(xtrain_res, ytrain_res)
# print("Best MLP Parameters: ", mlp_grid.best_params_)

# Train with the best parameters and fit to resmapled dataset
best_act = 'relu'
best_alpha = 1
best_rate = 'constant'
best_solver = 'adam'
mlp_best = MLPClassifier(hidden_layer_sizes=(20,10),
                         activation=best_act,
                         alpha=best_alpha,
                         learning_rate=best_rate,
                         solver=best_solver,
                         random_state=17)
mlp_best.fit(xtrain, ytrain)

# Report performance
y_pred_mlp = mlp_best.predict(xtest)
print("\nNeural Networks Performance: ")
plot_cm(ytest, y_pred_mlp, title='Confusion Matrix for Neural Networks')
print(classification_report(ytest, y_pred_mlp))

# ---------- Comparison ----------
predictions = [y_pred_knn, y_pred_logreg, y_pred_svm, y_pred_ksvm, y_pred_mlp]
models = ["KNN", "LogReg", "SVM", "Kernel SVM", "Neural Networks"]
accuracy = []
precision = []
recall = []
f1 = []
for prediction in predictions:
    accuracy.append(accuracy_score(ytest, prediction))
    precision.append(precision_score(ytest, prediction, average=None))
    recall.append(recall_score(ytest, prediction, average=None))
    f1.append(f1_score(ytest, prediction, average=None))

# Accuracies
plt.bar(models, accuracy, color='steelblue')
for m in range(len(models)):
    plt.text(m, accuracy[m], str(accuracy[m]), ha='center', va='bottom')
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

# Precisions
for m in range(len(models)):
    plt.plot(np.arange(10), precision[m], label=models[m])
plt.title("Precision Comparison")
plt.ylabel("Precision")
plt.xlabel("Classes")
plt.legend()
plt.show()

# Recalls
for m in range(len(models)):
    plt.plot(np.arange(10), recall[m], label=models[m])
plt.title("Recall Comparison")
plt.ylabel("Recall")
plt.xlabel("Classes")
plt.legend()
plt.show()

# F1-s
for m in range(len(models)):
    plt.plot(np.arange(10), f1[m], label=models[m])
plt.title("F1-score Comparison")
plt.ylabel("F1-score")
plt.xlabel("Classes")
plt.legend()
plt.show()