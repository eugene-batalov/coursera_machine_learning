# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.cross_validation import KFold, cross_val_score
import datetime
from sklearn.metrics import roc_auc_score


def auc_roc(estimator, X, y):
    clf = estimator
    clf.fit(X, y)
    pred = clf.predict_proba(X)[:, 1]
    return roc_auc_score(y, pred)

def est(n):
    return ensemble.GradientBoostingClassifier(random_state=1, n_estimators=n)

def cross_val_score_n(n, X, y, kf):
    start_time = datetime.datetime.now()
    cvs = cross_val_score(est(n), X, y=y, cv=kf, scoring=auc_roc)#'roc_auc')
    print("Trees: ", n, " Quality = ", np.mean(cvs),  " cross_val_score done in ", datetime.datetime.now() - start_time)
    return cvs
    
# Load data
data= pd.read_csv('features.csv', index_col='match_id')
X_test = pd.read_csv('features_test.csv', index_col='match_id')

#finish = list(set(data.columns.values) - set(data_test.columns.values))
X = data[list(X_test.columns.values)]

# Q1
missing_data = len(X) - X.count()
missing_data = missing_data[missing_data > 0]
print("Columns with missing values: ", missing_data.index.values)

# Q1 Alternative
X_full = X.dropna(axis=1, how='any')
missing_cols = list(set(X.columns.values) - set(X_full.columns.values))
#print(Columns with missing values: ", missing_cols)

X = X.fillna(value = 0)

# Q2
y = data['radiant_win']
X_test = X_test.fillna(value=0)
print("Target column name: ", y.name)

# Q3
start_time = datetime.datetime.now()
clf = ensemble.GradientBoostingClassifier(random_state=1, n_estimators=30)#, verbose = True)
clf.fit(X, y)
pred = clf.predict_proba(X)[:, 1]
print ('Time elapsed: ', datetime.datetime.now() - start_time, ' quality: ', roc_auc_score(y, pred))

# Q4
kf = KFold(len(y), n_folds = 5, shuffle=True, random_state=1)
opt = list(map(lambda x: np.mean(cross_val_score_n(x, X, y, kf)),(10,20,30,40,50,70)))

print(opt)
