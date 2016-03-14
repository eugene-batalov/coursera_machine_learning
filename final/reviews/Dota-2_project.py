#подход 1
#1
import pandas
import numpy as np
features = pandas.read_csv('./features.csv', index_col='match_id')

features.head()
f_len = len(features.columns)
features_ = features.drop(features.columns[(f_len-4):f_len], axis=1)
features_ = features_.drop(features.columns[(f_len-6)], axis=1)

features_test = pandas.read_csv('./features_test.csv', index_col='match_id')
features_test.head()

#2
features_.isnull().sum()
len(features_.index)-features_.count()

#3
df = features_.fillna(0)
X_test = features_test.fillna(0)

#4
X_train = df.drop('radiant_win',1)
y_train = df['radiant_win']

#5
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score

import time
import datetime

# перебор количества деревьев
k_range = [10,20,30,40,50]
k_scores = []
for k in k_range:
    kf = KFold(y_train.size, n_folds=5, shuffle=True, random_state = 241)
    clf = GradientBoostingClassifier(n_estimators=k, verbose=True,random_state = 241)
    scores = cross_val_score(clf, X_train, y_train, cv=kf, scoring='roc_auc')
    k_scores.append(scores.mean())

print (k_scores)

#оценка временных затрат (30 деревьев)
start_time = datetime.datetime.now()

kf = KFold(y_train.size, n_folds=5, shuffle=True, random_state = 241)
clf = GradientBoostingClassifier(n_estimators=30, verbose=True,random_state = 241)
scores = cross_val_score(clf, X_train, y_train, cv=kf, scoring='roc_auc')
scores.mean()

#качество кросс-валидации
#0.68949620394147093

print ('Time elapsed:', datetime.datetime.now() - start_time)

#время настройки классификатора
#Time elapsed: 0:01:34.022361



#подход 2
#1
#масштабирование признаков
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
sX_train = scaler.fit_transform(X_train)
sX_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn import grid_search

#выбор параметра регуляризации C (логистическая регрессия)
grid = {'C': np.power(10.0, np.arange(-5, 5))}
kf = KFold(y_train.size, n_folds=5, shuffle=True, random_state=241)
clf = LogisticRegression(penalty='l2',verbose=True,random_state = 241)
gs = grid_search.GridSearchCV(clf, grid, scoring='roc_auc', cv=kf)
gs.fit(sX_train, y_train)
for a in gs.grid_scores_:
    a.mean_validation_score
    a.parameters

#оценка временных затрат (C = 0.01)
start_time = datetime.datetime.now()

kf = KFold(y_train.size, n_folds=5, shuffle=True, random_state=241)
clf = LogisticRegression(C = 0.01, penalty='l2',verbose=True,random_state = 241)
scores_log = cross_val_score(clf, sX_train, y_train, cv=kf, scoring='roc_auc')
scores_log.mean()

#качество кросс-валидации
#0.71634146536541743

print ('Time elapsed:', datetime.datetime.now() - start_time)

#время настройки классификатора
#Time elapsed: 0:00:30.673725

#2
# удаление категориальных признаков
X_train1 = X_train.drop(X_train.columns[1],axis=1)
X_test1 = X_test.drop(X_test.columns[1],axis=1)
for i in range(0,9):
	X_train1 = X_train1.drop(X_train1.columns[1+8*i],axis=1)
	X_test1 = X_test1.drop(X_test1.columns[1+8*i],axis=1)

sX_train1 = scaler.fit_transform(X_train1)
sX_test1 = scaler.transform(X_test1)

#оценка параметра регуляризации
gs.fit(sX_train1, y_train)
for a in gs.grid_scores_:
    a.mean_validation_score
    a.parameters

#оценка временных затрат (C = 0.01)
start_time = datetime.datetime.now()

kf = KFold(y_train.size, n_folds=5, shuffle=True, random_state=241)
clf = LogisticRegression(C = 0.01, penalty='l2',verbose=True,random_state = 241)
scores_log1 = cross_val_score(clf, sX_train1, y_train, cv=kf, scoring='roc_auc')
scores_log1.mean()

#качество кросс-валидации
#0.71562271890831552

print ('Time elapsed:', datetime.datetime.now() - start_time)

#время настройки классификатора
#Time elapsed: 0:00:36.572557

#3
# поиск уникальных героев
u = np.unique(X_train[X_train.columns[[2+8*i for i in range(0,10)]].values.tolist()])

#4
# формирование мешка слов

X_pick = np.zeros((X_train.shape[0], max(u)))

for i, match_id in enumerate(X_train.index):
    for p in xrange(5):
        X_pick[i, X_train.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, X_train.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

X_pick_test = np.zeros((X_test.shape[0], max(u)))

for i, match_id in enumerate(X_test.index):
    for p in xrange(5):
        X_pick_test[i, X_test.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick_test[i, X_test.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

# добавление в исходный датасет
X_train2 = np.concatenate([X_train1, X_pick], axis=1)
X_test2 = np.concatenate([X_test1, X_pick_test], axis=1)

sX_train2 = scaler.fit_transform(X_train2)
sX_test2 = scaler.transform(X_test2)

#5
gs.fit(sX_train2, y_train)
for a in gs.grid_scores_:
    a.mean_validation_score
    a.parameters

#оценка временных затрат (C = 0.01)
start_time = datetime.datetime.now()

kf = KFold(y_train.size, n_folds=5, shuffle=True, random_state=241)
clf = LogisticRegression(C = 0.01, penalty='l2',verbose=True,random_state = 241)
scores_log1 = cross_val_score(clf, sX_train2, y_train, cv=kf, scoring='roc_auc')
scores_log1.mean()

#качество кросс-валидации
#0.75087711671101576

print ('Time elapsed:', datetime.datetime.now() - start_time)

#время настройки классификатора
#Time elapsed: 0:00:27.600690

#6
#прогноз на тестовой выборке
clf = LogisticRegression(C = 0.01, penalty='l2',verbose=True,random_state = 241)
clf.fit(sX_train2, y_train)

pred_log2 = clf.predict_proba(sX_test2)[:, 1]

min(pred_log2)
#0.0085207431136441088
max(pred_log2)
#0.99670462203367871§

