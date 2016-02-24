import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

data = pd.read_csv('gbm-data.csv')
np_data = data.values
X = np_data[:,1:]
y = np_data[:,[0]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

#print(X, y)