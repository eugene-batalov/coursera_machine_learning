import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import ensemble


data = pd.read_csv('gbm-data.csv')
np_data = data.values
X = np_data[:,1:]
y = np_data[:,[0]].ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

learning_rates = [1, 0.5, 0.3, 0.2, 0.1] 
clf = ensemble.GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241)
clf.fit(X_train, y_train)

#print(X, y)