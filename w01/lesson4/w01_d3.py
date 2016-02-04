# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 07:40:49 2016

@author: evgeny batalov
"""

import pandas
import graphviz
import numpy as np
import sklearn.tree as st



data = pandas.read_csv('titanic.csv', index_col='PassengerId')
data = data[['Pclass','Fare','Age','Sex','Survived']].dropna()
survived = data['Survived']
data = data[['Pclass','Fare','Age','Sex']]
data['Sex'] = data['Sex'].apply(lambda x: 1.0 if 'male' == x else 0.0)
X = np.array(data)
y = np.array(survived)
clf = st.DecisionTreeClassifier(random_state = 241)
clf.fit(X, y)
imp = np.around(clf.feature_importances_, decimals = 2)
#print(data['Survived'])

X1 = clf.predict(data[['Pclass','Fare','Age','Sex']])
print(X1.shape)

with open('tree.dot', 'w') as dotfile:
   st.export_graphviz(
        clf,
        dotfile,
        feature_names=['Pclass','Fare','Age','Sex','Survived'])
