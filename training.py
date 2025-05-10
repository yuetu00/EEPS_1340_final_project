#ridge regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

data_path_train_save = "Dataframe/all_test.csv"
df = pd.read_csv(data_path_train_save)

#0 is none
#1 is old_crater
#2 ejecta
df['binary_class'] = df['classification'].map(lambda x: 
                                                0 if x in ['none']
                                                else 1 if x in ['old_crater']
                                                else 2 
                                              )



# PCA + Logistic Regression
pca = PCA(n_components=5) 
features = df.drop(columns=['classification', 'Unnamed: 0'])
pca_result = pca.fit_transform(features)

#add pca componenets to the dataframe
df['pca1'] = pca_result[:, 0]
df['pca2'] = pca_result[:, 1]


#splitting data
X = df.drop(columns=['classification', 'Unnamed: 0', 'binary_class'])
y = df['binary_class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#feature processing after splitting data only with training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) #fit only on training
X_test_scaled = scaler.transform(X_test)    #transform test set

# KNN

# These commented out lines of code were used for identifying the optimal K value

# knn = KNeighborsClassifier()
# param_grid = {'n_neighbors': range(1, 21)}

# grid_search = GridSearchCV(knn, param_grid, cv=5)
# grid_search.fit(X_train, y_train)

# optimal_k = grid_search.best_params_['n_neighbors']
# print(f"Optimal K value: {optimal_k}")

knn = KNeighborsClassifier(n_neighbors=12).fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("KNN Classification Report:")
print(classification_report(y_test, y_pred))

print("KNN Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# used when we were conducting binary classification;
# not relevant for our updating classification technique

# print("PCA + Logistic Regression Classification Report:")
# print(classification_report(y_test, y_pred))

# print("PCA + Logistic Regression Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# Random Forest
X = df.drop(columns=['classification', 'Unnamed: 0', 'binary_class'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# param_grid = {'n_estimators': [100, 200, 500],
#               'max_depth': [None, 10, 20],
#               'max_features': ['sqrt', 'log2']
# }

# grid_search = GridSearchCV(RandomForestClassifier(class_weight='balanced'),
#                            param_grid=param_grid).fit(X_train, y_train)
# print("Best parameters: ", grid_search.best_params_)

rf = RandomForestClassifier(n_estimators=200, max_depth=30, max_features="sqrt", class_weight="balanced", random_state=2025).fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred))

print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# SVM

# param_grid = {
#     'C': [0.1, 1, 10, 20, 50, 70],
#     'gamma': ['scale', 0.1, 0.01, 0.001],
#     'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
# }

# grid_search = GridSearchCV(SVC(class_weight='balanced'),
#                            param_grid=param_grid).fit(X_train, y_train)
# print("Best parameters: ", grid_search.best_params_)

svm = SVC(C=50, gamma=0.01, kernel='rbf', class_weight='balanced').fit(X_train, y_train)
y_pred = svm.predict(X_test)

print("SVM Classification Report:")
print(classification_report(y_test, y_pred))

print("SVM Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_train, y_train)
y_pred = dt.predict(X_test)

print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred))

print("Decision Tree SVM Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Adaboost

# param_grid_lr = {'n_estimators': [100, 200, 500],
#               'learning_rate': [0.1, 1.0, 10.0, 20.0],
#               'estimator__C': [0.1, 1.0, 10.0]
# }

# Adaboost with Logistic Regression 
estimator = logreg_ridge = LogisticRegression(C=10.0, penalty='l2', max_iter=1000)
adaboost = AdaBoostClassifier(base_estimator=estimator, n_estimators=100, learning_rate=0.1, random_state=42)

# grid_search = GridSearchCV(adaboost, param_grid_lr, scoring='accuracy', cv=5).fit(X_train, y_train)
# print("Best parameters: ", grid_search.best_params_)

print("AdaBoost + Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred))

print("AdaBoost + Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Adaboost with Decision Tree

# param_grid_dt = {
#     'estimator__max_depth': [1, 2, 3, 5, 10],
#     'n_estimators': [50, 100],
#     'learning_rate': [0.01, 0.1, 1.0]
# }

estimator = decision_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
adaboost = AdaBoostClassifier(estimator=estimator, n_estimators=100, learning_rate=1.0, random_state=42).fit(X_train, y_train)

# grid_search = GridSearchCV(adaboost, param_grid_dt, scoring='accuracy', cv=5).fit(X_train, y_train)
# print("Best parameters: ", grid_search.best_params_)

print("AdaBoost + Decision Tree Classification Report:")
print(classification_report(y_test, y_pred))

print("AdaBoost + Decision Tree Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))