# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.cross_validation import KFold, cross_val_score
import datetime
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn import grid_search

def time_interval(delta):
    s = delta.split(':')
    return s[0]+ ' часов ' + s[1]+ ' минут ' + str(np.round(float(s[2]), 2)) + ' секунд '
    
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
    time = time_interval(str(datetime.datetime.now() - start_time))
    print("Деревьев: ", n, " Качество = ", np.mean(cvs),  " cross_val_score выполнено за: ", time)
    return np.mean(cvs), time
    
# Чтение данных
data= pd.read_csv('features.csv', index_col='match_id')
X_test = pd.read_csv('features_test.csv', index_col='match_id')

#finish = list(set(data.columns.values) - set(data_test.columns.values))
X = data[list(X_test.columns.values)]

# Часть 1
# Вопрос 1
missing_data = len(X) - X.count()
missing_data = missing_data[missing_data > 0]
print("Имена столбцов с пропущенными данными: \n", missing_data.index.values)
print(
"""Для столбца first_blood_team пропуск означает, что в первые 5 минут ни один герой не погиб.
Для столбца first_blood_player2 (кто из игроков совершил первое убийство) пропуск может означать, 
что или никто не погиб в первые 5 минут, или первый игрок погиб от крипов или башни."""
)

# Вопрос 1 альтернатива
X_full = X.dropna(axis=1, how='any')
missing_cols = list(set(X.columns.values) - set(X_full.columns.values))
#print(Columns with missing values: ", missing_cols)

X = X.fillna(value = 0)
X_test = X_test.fillna(value=0)

# Вопрос 2
y = data['radiant_win']
print("Столбец, содержащий целевую переменную называется: ", y.name)

# Вопросы 3, 4
kf = KFold(len(y), n_folds = 5, shuffle=True, random_state=1)
#opt = list(map(lambda x: cross_val_score_n(x, X, y, kf),(10,20,30,40,50,70)))
#print(opt)
#print(
#"""Кросс-валидация для градиентного бустинга с 30 деревьями проводилась: """, opt[2][1],
#""" оценка качества получилась: """, opt[2][0],
"""\nПри увеличении количества деревьев качество растет, имеет смысл их увеличивать.
Для ускорения просчета можно попробовать взять меньше обучающих данных, а также
попробовать поиграть параметрами GradientBoostingClassifier - learning_rate,
max_depth или max_leaf_nodes, max_features, а также попробовать другие метрики
для кросс-валидации."""
#)

# Часть вторая
X_s = StandardScaler().fit_transform(X)

def cross_val_score_c(c, X, y, kf):
    start_time = datetime.datetime.now()
    clf = linear_model.LogisticRegression(C=c, penalty='l2',random_state = 241)
    clf.fit(X, y)
    cvs = cross_val_score(clf, X, y=y, cv=kf, scoring=auc_roc)
    time = time_interval(str(datetime.datetime.now() - start_time))
    return np.mean(cvs), time, c

# Вопрос 1   

# Вариант
#С_val = np.power(10.0, np.arange(-5, 6)) 
#opt = list(map(lambda x: cross_val_score_c(x, X_s, y, kf), С_val))
#c_arr = []
#for o in opt:
#    c_arr.append(o[2])
#    print('Время выполнения: ', o[1], ' Качество: ', o[0], ' c= ', o[2])
#c = С_val[np.argmax(c_arr)]
#print("""Качество логистической регрессии на этой задаче сопоставимо с градиентным бустингом,
#время выполнения значительно ниже, параметр С влияет в основном на время выполнения, нименьшее
#время при С = """, c)
grid = {'C': np.power(10.0, np.arange(-5, 5))}
clf = linear_model.LogisticRegression(penalty='l2',random_state = 241)
gs = grid_search.GridSearchCV(clf, grid, scoring='roc_auc', cv=kf)
gs.fit(X_s, y)
scores = gs.grid_scores_
print('scores: ', scores, ' best esmitator: ', gs.best_estimator_)
c = 0.01

# Вопрос 2
categorial_columns = [s for s in X.columns.values if 'hero' in s or 'lobby_type' in s]
X_nc = X.drop(categorial_columns, 1)
X_s = StandardScaler().fit_transform(X_nc)
cvs = cross_val_score_c(c, X_s, y, kf)
print("После удаления категорийных столбцов: ", 'Время выполнения: ', cvs[1], ' Качество: ', cvs[0])

# Вопрос 3
heroes = pd.read_csv('heroes.csv', index_col='id')
N = len(heroes)
print("Количество героев в игре: ", N)
X_pick = np.zeros((data.shape[0], N))

# Вопрос 4
for i, match_id in enumerate(data.index):
    for p in range(0,5):
        X_pick[i, data.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, data.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
X_h = X_nc.join(pd.DataFrame(X_pick, index=data.index), how = 'right')
X_s = StandardScaler().fit_transform(X_h) 
cvs = cross_val_score_c(c, X_s, y, kf)
print("После добавления мешка героев: ", 'Время выполнения: ', cvs[1], ' Качество: ', cvs[0])
print("""Качество предсказания значительно выросло, вероятно использование определенных наборов героев
существенно влияет на шансы выигрыша той или иной команды""")

# Вопрос 5
clf = linear_model.LogisticRegression(C=c, penalty='l2')
clf.fit(X_s, y)

X_pick_test = np.zeros((X_test.shape[0], N))

for i, match_id in enumerate(X_test.index):
    for p in range(0,5):
        X_pick_test[i, X_test.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick_test[i, X_test.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
X_nc_test = X_test.drop(categorial_columns, 1)
X_h_test = X_nc_test.join(pd.DataFrame(X_pick_test, index=X_test.index), how = 'right')
X_s_test = StandardScaler().fit_transform(X_h_test)

pred = clf.predict_proba(X_s_test)[:, 1]
print("Значения прогноза на тестовой выборке, минимальное: ", np.min(pred)," максимальное: ", np.max(pred))