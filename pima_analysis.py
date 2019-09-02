# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 10:42:17 2019

@author: Colton Smith
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

df = pd.read_csv('pima.csv')

# plt.hist(df.Skin)

df.Label = np.where(df.Label == 'tested_positive', 1, 0)
# plt.hist(df.Label)

corr_matrix = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


### Model Building ###

y = df.Label
X = df.drop('Label', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

### Regularization ###
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

clf = LogisticRegression(random_state=1, solver='liblinear', penalty = 'l1', C = 0.1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)
baseline_accuracy = len(y[y == 0]) / (len(y))
cohens_score = cohen_kappa_score(y_test, y_pred)

ind = range(0, clf.coef_.shape[1])
plt.bar(x = ind, height = clf.coef_[0])
plt.xticks(ind, X.columns)

###############################################################################

fold_perf = []

kf = KFold(n_splits=4, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc_score = accuracy_score(y_test, y_pred)
    fold_perf.append(acc_score)
    
print(np.mean(fold_perf))

hyperparam_grid = {'n_estimators': [3, 100, 1000],
                   'max_features': [0.05, 0.5, 0.95],
                   'max_depth': [10, 50, None]}

grid_scorer = make_scorer(cohen_kappa_score)
clf = GridSearchCV(RandomForestClassifier(), hyperparam_grid, cv = kf, scoring = grid_scorer)
clf.fit(X, y)

print(clf.best_score_)
print(clf.best_params_)