# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 07:40:49 2016

@author: galinabatalova
"""

import pandas
import itertools
data = pandas.read_csv('titanic.csv', index_col='PassengerId')
sex = data['Sex'].value_counts()

print("1.", sex[0], sex[1])

survived =  data['Survived'].value_counts()

print('2.', round(survived[1]/sum(survived)*100,2))

pclass = data['Pclass'].value_counts()

print('3.',round(pclass[1]/sum(pclass)*100,2))

age = data['Age']

age = age[~age.isnull()]

print('4.', round(age.mean(),2), round(age.median(),2))

s = data['SibSp']

p = data['Parch']

toCorr=pandas.concat([data['SibSp'],data['Parch']], axis = 1)

corr=toCorr.corr(method='pearson')

print('5.', round(corr['SibSp']['Parch'],2))

women = data[data['Sex'] == 'female']
women = women['Name']
mrs=women[women.str.contains('Mrs.')]# | women.str.contains('Mme.')]
ms=women[women.str.contains('Miss.')]# | women.str.contains('Mlle.') | women.str.contains('Ms.')]
other=women[~women.str.contains('Mrs.') & ~women.str.contains('Miss.')]
wFirstNamesMrs=mrs.str.split('(').str[1].dropna().str.replace(')','').str.replace('"','').str.split(' ')
wFirstNamesMs=ms.str.split('Miss. ').str[1].str.replace('"','').str.split(' ')
wFirstNamesMs=wFirstNamesMs.values.tolist()
wFirstNamesMs=list(itertools.chain(*wFirstNamesMs))
wFirstNamesMrs=wFirstNamesMrs.values.tolist()
wFirstNamesMrs=list(itertools.chain(*wFirstNamesMrs))
wFirstNames = wFirstNamesMrs + wFirstNamesMs
wFirstNames = pandas.DataFrame(wFirstNames, columns=['Name'])
wFirstNames = wFirstNames['Name'].value_counts()
print('6.', wFirstNames.index[0])