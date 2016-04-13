#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from __future__ import print_function, division
import json
import bz2
import operator
import math

import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
from sklearn import grid_search

import extract_features  as ef

import numpy as np
import pandas

from sklearn.preprocessing import StandardScaler

"""
1. Какие признаки имеют пропуски среди своих значений (приведите полный список имен этих признаков)? 
Что могут означать пропуски в этих признаках (ответьте на этот вопрос для двух любых признаков)?
['first_blood_time', 'first_blood_team', 'first_blood_player1', 'first_blood_player2', 'radiant_bottle_time', 'radiant_courier_time', 'radiant_flying_courier_time', 'radiant_first_ward_time', 'dire_bottle_time', 'dire_courier_time', 'dire_flying_courier_time', 'dire_first_ward_time']

first_blood_time = за данный отрезок игры никто ещё не погиб
first_blood_team = посколько никто ещё не пролил кровь, то эти данные отсутствуют

2. Как называется столбец, содержащий целевую переменную?
radiant_win

3. Как долго проводилась кросс-валидация для градиентного бустинга с 30 деревьями? Инструкцию по измерению времени можно найти выше по тексту. Какое качество при этом получилось?
Для одного конкретного набора параметров (learning_reate, n_estimators) - 
порядка 4 минут

Для логрегрессии - порядка минуты двух на каждое C - но, разумеется, это всё зависит
от количества конкретных параметров. 

4. Имеет ли смысл использовать больше 30 деревьев в градиентном бустинге? Что можно сделать, чтобы ускорить его обучение при увеличении количества деревьев?
Из того, что я вижу - score  улушчается до приблизительно n_estimators = 150  
после чего начинает "пробуксовка".
Использовать больше деревьев стоит, но не на порядки.
C какого-то момента будет бессмысленно добавлять деревья - глубина данных
такова что никаких новых результатов мы не получим

Гораздо эффективней придумывать правила и генерировать фичи - и эффективней
и интересней)

==
1. Какое качество получилось у логистической регрессии над всеми исходными признаками? Как оно соотносится с качеством градиентного бустинга? Чем вы можете объяснить эту разницу? Быстрее ли работает логистическая регрессия по сравнению с градиентным бустингом?
(Отсортировано по убыванию скора)
SCORES  [(0.71627287010351204, {'C': 1}), (0.71627232990537837, {'C': 10}), (0.71627231189193397, {'C': 100}), (0.71627228435354451, {'C': 1000}), (0.71627228117872277, {'C': 10000})]

2. Как влияет на качество логистической регрессии удаление категориальных признаков (укажите новое значение метрики качества)? 
Чем вы можете объяснить это изменение? 
Практически не влияет:

SCORES  [(0.71630563327436403, {'C': 1}), (0.71630525843655268, {'C': 10}), (0.71630516737632577, {'C': 1000}), (0.71630516418677348, {'C': 100}), 
(0.71630515784156912, {'C': 10000})]

Объясняется это должно быть тем что логрегрессия сама их каким-то образом минимизировала

3. Сколько различных идентификаторов героев существует в данной игре?
THERE ARE 112 UNIQUE HEROES

4. Какое получилось качество при добавлении "мешка слов" по героям? Улучшилось ли оно по сравнению с предыдущим вариантом? Чем вы можете это объяснить?
SCORES  [(0.75174702099484403, {'C': 1}), (0.75174624572446125, {'C': 10}), (0.75174618749923328, {'C': 1000}), (0.75174617479129624, {'C': 10000}), (0.75174617266925436, {'C': 100})]

5. Какое минимальное и максимальное значение прогноза на тестовой выборке получилось у лучшего из алгоритмов? 
0.712914316236 
0.75174702099484403


Thank you and have a good day! 
"""

def get_boosting_classifier(params):
    (kfolds) = params
    clf = GradientBoostingClassifier(verbose=True, random_state=241)
    param_grid = [
            {'n_estimators': [10], 'learning_rate': [0.5, 0.55]}
     ]
    gs = grid_search.GridSearchCV(clf, param_grid, scoring='roc_auc', cv=kfolds)
    return gs

def get_logreg_classifier(params):
    (kfolds) = params
    clf = sklearn.linear_model.LogisticRegression(penalty='l2', random_state=241)
    param_grid = [
        {'C': [0.001, 0.01, 0.1, 0.2, 0.5, 0.55,  1]}
     ]
    gs = grid_search.GridSearchCV(clf, param_grid, scoring='roc_auc', cv=kfolds)
    return gs

def get_SVC_classifier(params):
    (C) = params
    clf = sklearn.svm.SVC(kernel='linear', C=C, random_state=241)
    return clf

def get_score(X, y, clf):
    # This one I've used only to check do I'm doing it right with grid search
    kfolds = KFold(X.shape[0], shuffle=True, n_folds=5, random_state=1)
    scores = sorted(sklearn.cross_validation.cross_val_score(
        clf,
        X, y,
        cv=kfolds,
        scoring='roc_auc'
    ))
    score = np.mean(scores)
    return score


def create_table(in_fname, out_fname):
    time_point = 5 * 60
    features_table = ef.create_table(in_fname, time_point)
    features_table.to_csv(out_fname)
    
def transform_simple(data):
    data.fillna(0, inplace = True)
    return data

def transform_omit_categorial(data):
    data.fillna(0, inplace = True)
    data.drop('lobby_type', axis=1, inplace=True)
    data.drop('r1_hero', axis=1, inplace=True)
    data.drop('r2_hero', axis=1, inplace=True)
    data.drop('r3_hero', axis=1, inplace=True)
    data.drop('r4_hero', axis=1, inplace=True)
    data.drop('r5_hero', axis=1, inplace=True)
    data.drop('d1_hero', axis=1, inplace=True)
    data.drop('d2_hero', axis=1, inplace=True)
    data.drop('d3_hero', axis=1, inplace=True)
    data.drop('d4_hero', axis=1, inplace=True)
    data.drop('d5_hero', axis=1, inplace=True)
    return data

def transform_data_with_bag(df):
    #df = transform_data(df)

    N = 112 #just passed through set and computed"
    X_pick = np.zeros((df.shape[0], N))

    for i, match_id in enumerate(df.index):
        for p in xrange(5):
            X_pick[i, df.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
            X_pick[i, df.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

    xpick_df = pandas.DataFrame(X_pick)
    xpick_df = xpick_df.astype(int)
    xpick_df.index = df.index

    df = df.join(xpick_df, how="inner")
    df = transform_omit_categorial(df)

    df.fillna(0, inplace = True)
    return df

def transform_data(data):
    r_kills = data["r1_kills"] + data["r2_kills"] + data["r3_kills"] + data["r4_kills"] + data["r5_kills"]
    d_kills = data["d1_kills"] + data["d2_kills"] + data["d3_kills"] + data["d4_kills"] + data["d5_kills"]
    r_deaths = data["r1_deaths"] + data["r2_deaths"] + data["r3_deaths"] + data["r4_deaths"] + data["r5_deaths"]
    d_deaths = data["d1_deaths"] + data["d2_deaths"] + data["d3_deaths"] + data["d4_deaths"] + data["d5_deaths"]

    total_kills = (r_kills + d_kills)
    rel_r_kills = r_kills / total_kills
    rel_r_kills.fillna(-1, inplace = True)

    total_deaths = r_deaths + d_deaths
    rel_r_deaths = r_deaths / total_deaths
    rel_r_deaths.fillna(-1, inplace = True)

    #data['rel_r_kills'] = rel_r_kills
    #data['rel_r_deaths'] = rel_r_deaths
    r_level = (data["r1_level"] + data["r2_level"] + data["r3_level"] + data["r4_level"] + data["r5_level"])/5
    d_level = (data["d1_level"] + data["d2_level"] + data["d3_level"] + data["d4_level"] + data["d5_level"])/5

    r_gold = (data["r1_gold"] + data["r2_gold"] + data["r3_gold"] + data["r4_gold"] + data["r5_gold"])
    d_gold = (data["d1_gold"] + data["d2_gold"] + data["d3_gold"] + data["d4_gold"] + data["d5_gold"])

    gold_diff = (r_gold / d_gold) 
    data['gold_diff'] = gold_diff

    data.drop('r1_gold', axis=1, inplace=True)
    data.drop('r2_gold', axis=1, inplace=True)
    data.drop('r3_gold', axis=1, inplace=True)
    data.drop('r4_gold', axis=1, inplace=True)
    data.drop('r5_gold', axis=1, inplace=True)
    data.drop('d1_gold', axis=1, inplace=True)
    data.drop('d2_gold', axis=1, inplace=True)
    data.drop('d3_gold', axis=1, inplace=True)
    data.drop('d4_gold', axis=1, inplace=True)
    data.drop('d5_gold', axis=1, inplace=True)

    r_xp = (data["r1_xp"] + data["r2_xp"] + data["r3_xp"] + data["r4_xp"] + data["r5_xp"])
    d_xp = (data["d1_xp"] + data["d2_xp"] + data["d3_xp"] + data["d4_xp"] + data["d5_xp"])


    xp_diff = (r_xp / d_xp) 
    data['xp_diff'] = xp_diff
    data.drop('r1_xp', axis=1, inplace=True)
    data.drop('r2_xp', axis=1, inplace=True)
    data.drop('r3_xp', axis=1, inplace=True)
    data.drop('r4_xp', axis=1, inplace=True)
    data.drop('r5_xp', axis=1, inplace=True)
    data.drop('d1_xp', axis=1, inplace=True)
    data.drop('d2_xp', axis=1, inplace=True)
    data.drop('d3_xp', axis=1, inplace=True)
    data.drop('d4_xp', axis=1, inplace=True)
    data.drop('d5_xp', axis=1, inplace=True)

    r_lh = (data["r1_lh"] + data["r2_lh"] + data["r3_lh"] + data["r4_lh"] + data["r5_lh"])
    d_lh = (data["d1_lh"] + data["d2_lh"] + data["d3_lh"] + data["d4_lh"] + data["d5_lh"])

    data.drop('r1_lh', axis=1, inplace=True)
    data.drop('r2_lh', axis=1, inplace=True)
    data.drop('r3_lh', axis=1, inplace=True)
    data.drop('r4_lh', axis=1, inplace=True)
    data.drop('r5_lh', axis=1, inplace=True)
    data.drop('d1_lh', axis=1, inplace=True)
    data.drop('d2_lh', axis=1, inplace=True)
    data.drop('d3_lh', axis=1, inplace=True)
    data.drop('d4_lh', axis=1, inplace=True)
    data.drop('d5_lh', axis=1, inplace=True)

    lh_diff = (r_lh - d_lh) 
    data['lh_diff'] = lh_diff

    data.fillna(0, inplace = True)
    return data



def dota_predict(get_params, get_classifier, transform_data, get_score = get_score):
    Y_name = 'radiant_win'
    finish_features = set([
        'duration', 
        'tower_status_radiant', 
        'tower_status_dire',
        'barracks_status_dire',
        'barracks_status_radiant',
        Y_name
    ])

    features_file = "matches.csv"
    test_file = "matches_test.csv"

    #features_file = "artifacts/features.csv"
    #test_file = "artifacts/features_test.csv"

    data = pandas.read_csv(features_file, index_col='match_id')

    columns = [column for column in data.columns if column not in finish_features]
    filtered_data = data[columns]

    columns_with_missing_vals = [column for column in columns if (len(filtered_data)- filtered_data[column].count()) != 0]
    print(columns_with_missing_vals)

    X = transform_data(filtered_data)
    y = data[Y_name]
    total = len(X)


    params = get_params(X, y)
    clf = get_classifier(params)
    print("PARAMS ", params)
    #print("SCORE {}".format(get_score(X, y, clf)))

    standard_scaler = StandardScaler()
    #X = standard_scaler.fit_transform(X)
    X = standard_scaler.fit_transform(X)

    clf.fit(X, y)
    scores = sorted([(a.mean_validation_score, a.parameters) for a in  clf.grid_scores_], reverse=True)
    print("SCORES ", scores)

    data_test = transform_data(pandas.read_csv(test_file, index_col='match_id'))
    data_test_scaled = standard_scaler.transform(data_test)
    predicted = clf.predict_proba(data_test_scaled)


    prediction_df = pandas.DataFrame({
        'match_id': data_test.index.get_values(),
        'radiant_win': map(operator.itemgetter(1), predicted)
    })
    prediction_df.set_index('match_id', inplace=True);
    prediction_df.to_csv("result.csv")


def boosting():
    dota_predict(
                get_params = lambda X,y: (KFold(X.shape[0], shuffle=True, n_folds=5, random_state=1)),
                get_classifier = get_boosting_classifier,
                transform_data = transform
            )

def logreg(C=1.0):
    dota_predict(
                get_params = lambda X,y: (KFold(X.shape[0], shuffle=True, n_folds=5, random_state=1)),
                get_classifier = get_logreg_classifier,
                transform_data = transform_data_with_bag
            )

def svc(C=1.0):
    dota_predict(
                get_params = lambda X,y:(C),
                get_classifier = get_SVC_classifier,
                transform_data = transform_omit_categorial
            )

def main():
    #create_table('artifacts/data/matches.jsonlines.bz2', 'matches.csv')
    #create_table('artifacts/data/matches_test.jsonlines.bz2', 'matches_test.csv')
    #return
    # http://stackoverflow.com/questions/20625582/how-to-deal-with-this-pandas-warning
    pandas.options.mode.chained_assignment = None  
    #boosting()
    logreg()

    print("good to go! you can upload result.csv to kaggle.")
    #this one is better not to try )))
    #svc(1000.0)
    
if __name__ == '__main__':
    main()

