import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import graphviz
import seaborn as sns
np.random.seed(555)

# Load data file and set column names
with open('spambase.names', 'r') as f:
    lines = f.readlines()
namelines = [line for line in lines if ':' in line]
colnames = [line.split(':')[0] for line in namelines]
colnames = colnames[1:]
colnames.append('spam')
data = pd.read_csv('spambase.data', header=None, names=colnames)
data = data.fillna(0)

# Split labels and data
data_x = data.drop('spam', axis=1)
data_y = data['spam']


# ----- CART model -----

# 1. Train-test split
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=0)

# # 2. Parameter grid search set up
# grid = {'max_depth': [None, 5, 10, 15, 20],
#     'min_samples_split': [50, 100, 150, 200],
#     'min_samples_leaf': [10, 20, 30, 40],
#         'max_leaf_nodes': [None, 10, 20, 30, 40] }
#
# # 3. Grid Search with 5-fold CV
# cart = DecisionTreeClassifier(random_state=0)
# gridsearch = GridSearchCV(cart, grid, cv=5, verbose=2)
# gridsearch.fit(x_train, y_train)
#
# # 4. Report the best parameter combination
# best_params = gridsearch.best_params_
# print(f"Best CART parameters: {best_params}")

# After grid searchc the best params are:
best_params = {'max_depth': None, 'max_leaf_nodes': 20, 'min_samples_leaf': 10, 'min_samples_split': 50}

# 5. Train with best parameters
cart_best = DecisionTreeClassifier(**best_params, random_state=0)
cart_best.fit(x_train, y_train)

# 6. Predict and report accuracy
y_pred = cart_best.predict(x_test)
cart_acc = metrics.accuracy_score(y_test, y_pred)
print(f"CART accuracy: {cart_acc}")

# 7. Visualize the CART
dot_data = export_graphviz(cart_best, out_file=None,
                            feature_names=data_x.columns,
                            class_names=['Non-Spam','Spam'],
                            proportion=True,
                            filled=True,
                           rounded=True,
                           special_characters=True
                           )

graph = graphviz.Source(dot_data, format="png")
graph.render("decision_tree_graphivz")

# ----- Random Forest -----
# # 1. Parameter grid search set up
# grid_rf = {
#     # 'max_depth': [5, 10, 20, None],
#     'max_depth': [30, 50, 70, None],
#     # 'min_samples_split': [2, 5, 10, 20],
#     'min_samples_split': [2, 3, 5, 7, 10],
#     # 'min_samples_leaf': [2, 5, 10]
#     'min_samples_leaf': [1, 2, 3, 4, 5]}
#
# # 2. Grid Search with 5-fold CV
# rf = RandomForestClassifier(random_state=0)
# gridsearch_rf = GridSearchCV(rf, grid_rf, cv=5, verbose=2)
# gridsearch_rf.fit(x_train, y_train)
#
# # 3. Report the best parameter combination
# best_params_rf = gridsearch_rf.best_params_
# print(f"Best Random Forest parameters: {best_params_rf}")

# Grid search best params:
best_params_rf = {'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 2}

# 4. Train with best parameters
rf_best = RandomForestClassifier(**best_params_rf, random_state=0)
rf_best.fit(x_train, y_train)

# 5. Predict and report accuracy
y_pred_rf = rf_best.predict(x_test)
rf_acc = metrics.accuracy_score(y_test, y_pred_rf)
print(f"Random Forest accuracy: {rf_acc}")

# 6. Errors for both
cart_err = 1.0 - cart_acc
rf_err = 1.0 - rf_acc
print(f"CART test error rate: {cart_err}")
print(f"Random Forest test error rate: {rf_err}")

# 7. Several RForests for different # of trees
n_estimators_range = range(1, 502, 20) #reduced to 500, 1000 - in the report
rf_err_list = []
for n in n_estimators_range:
    rf = RandomForestClassifier(n_estimators=n, **best_params_rf, random_state=0)
    rf.fit(x_train, y_train)
    y_pred_rf = rf.predict(x_test)
    rf_err = 1.0 - metrics.accuracy_score(y_test, y_pred_rf)
    rf_err_list.append(rf_err)
    print(f"Completed for {n}")

# 8. Plot errors
plt.figure()
plt.plot(n_estimators_range, rf_err_list, label='Random Forest Test Error', color='steelblue')
plt.axhline(y=cart_err, color='orange', linestyle='-', label='CART Test Error')
plt.xlabel('Number of Trees')
plt.ylabel('Test Error Rate')
plt.title('Test Error VS Number of Trees')
plt.legend(loc='upper right')
plt.show()

# Displayed some of the errors for the report
# some_errors  = [1, 11, 31, 51, 76, 101, 201, 301, 401, 501, 751, 1001]
# err_d = {'Number of Trees': some_errors,
#             'Test Error': [rf_err_list[e//5] for e in some_errors]}
# some_errorsdf = pd.DataFrame(err_d)
# print(some_errorsdf)

# ----- Random Forest: tuning max_features -----

# 1. Set a range to try
ftrs = x_train.shape[1]
max_features_lst = list(range(1, ftrs+1))
rf_oob_errors = []
rf_test_errors = []

# 2. Experiment with max_features, the rest - tuned from last step
for mf in max_features_lst:
    rf = RandomForestClassifier(n_estimators=100, max_features=mf, #reduced to 100, in the report - 200
                                oob_score=True,
                                **best_params_rf, random_state=0)
    rf.fit(x_train, y_train)

    oob_error = 1 - rf.oob_score_
    rf_oob_errors.append(oob_error)

    y_pred_rf = rf.predict(x_test)
    test_error = 1.0 - metrics.accuracy_score(y_test, y_pred_rf)
    rf_test_errors.append(test_error)
    print(f"Completed for max_features: {mf}")

# 3. Plot the changes in OOB and Test error
# Top 5 lowest values for each curve
oob_mins = sorted(range(len(rf_oob_errors)), key=lambda sub: rf_oob_errors[sub])[:5]
test_mins = sorted(range(len(rf_test_errors)), key=lambda sub: rf_test_errors[sub])[:5]

plt.figure(figsize=(10, 5))
plt.scatter([max_features_lst[i] for i in oob_mins], [rf_oob_errors[i] for i in oob_mins], color='darkred')
plt.scatter([max_features_lst[i] for i in test_mins], [rf_test_errors[i] for i in test_mins], color='darkred')
plt.plot(max_features_lst, rf_oob_errors, label='OOB Error', color='steelblue')
plt.plot(max_features_lst, rf_test_errors, label='Test Error', color='orange')
plt.xlabel('max_features ($v$)')
plt.ylabel('Error Rate')
plt.title('Test Error Rate and OOB vs max_features ($v$)')
plt.xticks(np.arange(0, ftrs+1, 10))
plt.xticks(rotation=45)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()


# ----- One-class SVM -----
# 1. Scale, split, extract non-spams
sclr = StandardScaler()
data_xs = sclr.fit_transform(data_x)
xs_train, xs_test, y_train2, y_test2 = train_test_split(data_xs, data_y, test_size=0.2, random_state=1)
xs_train_notspam = xs_train[y_train2 == 0]

# # 2. Grid Search - bandwidth tuning
# grid_svm = {'gamma': [0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06]}
# svm = OneClassSVM(kernel='rbf')
# scoresvm = metrics.make_scorer(metrics.f1_score)
# gridsearch_svm = GridSearchCV(svm, grid_svm, cv=5, verbose=2, scoring=scoresvm)
# # Specify to svm that we are training on NON spam
# gridsearch_svm.fit(xs_train_notspam, [1]*len(xs_train_notspam))
#
# # 3. Report the best gamma
# best_params_svm = gridsearch_svm.best_params_
# print(f"Best One-Class SVM gamma: {best_params_svm}")

# From grid search we get:
best_params_svm = {'gamma': 0.035}

# 4. Train with rbf and best gamma
svm_best = OneClassSVM(kernel='rbf', **best_params_svm)
svm_best.fit(xs_train_notspam)

# 5. Make prediction and report performance
y_pred_svm = svm_best.predict(xs_test)
y_pred_svm = np.where(y_pred_svm == 1, 0, 1)

# 6. Error, CM, Results
svm_err = np.mean(y_pred_svm != y_test2)
print(f"One-Class SVM misclassification error: {svm_err}")

cm_svm = metrics.confusion_matrix(y_test2, y_pred_svm)
plt.figure()
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='BuPu',
                    xticklabels=['Non-Spam', 'Spam'],
                    yticklabels=['Non-Spam', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Original')
plt.title(f'Confusion Matrix One Class SVM,,\n Gamma: {best_params_svm["gamma"]}, \n Misclassification error: {round(svm_err*100,2)}%')
plt.show()

print(metrics.classification_report(y_test2,
                                    y_pred_svm,
                                    target_names=['Non-Spam', 'Spam']))

# ----- One-class SVM - tune NU as well -----
# 1. Scale, split, extract non-spams
sclr = StandardScaler()
data_xs = sclr.fit_transform(data_x)
xs_train, xs_test, y_train2, y_test2 = train_test_split(data_xs, data_y, test_size=0.2, random_state=1)
xs_train_notspam = xs_train[y_train2 == 0]

# # 2. Grid Search - bandwidth and nu tuning
# grid_svm = {'gamma': [0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06], 'nu': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]}
# svm = OneClassSVM(kernel='rbf')
# scoresvm = metrics.make_scorer(metrics.f1_score)
# gridsearch_svm = GridSearchCV(svm, grid_svm, cv=5, verbose=2, scoring=scoresvm)
# # Specify to svm that we are training on NON spam
# gridsearch_svm.fit(xs_train_notspam, [1]*len(xs_train_notspam))
#
# # 3. Report the best gamma and nu
# best_params_svm = gridsearch_svm.best_params_
# print(f"Best One-Class SVM parameters: {best_params_svm}")

# Grid search result:
best_params_svm = {'gamma': 0.03, 'nu': 0.001}

# 4. Train with rbf and best gamma and nu
svm_best = OneClassSVM(kernel='rbf', **best_params_svm)
svm_best.fit(xs_train_notspam)

# 5. Make prediction and report performance
y_pred_svm = svm_best.predict(xs_test)
y_pred_svm = np.where(y_pred_svm == 1, 0, 1)

# 6. Error, CM, Results
svm_err = np.mean(y_pred_svm != y_test2)
print(f"One-Class SVM misclassification error: {svm_err}")

cm_svm = metrics.confusion_matrix(y_test2, y_pred_svm)
plt.figure()
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='BuPu',
                    xticklabels=['Non-Spam', 'Spam'],
                    yticklabels=['Non-Spam', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Original')
plt.title(f'Confusion Matrix One Class SVM,\n Gamma: {best_params_svm["gamma"]}, nu: {best_params_svm["nu"]} \n Misclassification error: {round(svm_err*100,2)}%')
plt.show()

print(metrics.classification_report(y_test2,
                                    y_pred_svm,
                                    target_names=['Non-Spam', 'Spam']))