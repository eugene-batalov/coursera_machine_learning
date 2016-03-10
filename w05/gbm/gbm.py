import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn import ensemble
import matplotlib.pyplot as plt
from sklearn.ensemble.forest import RandomForestClassifier

learning_rates = [1, 0.5, 0.3, 0.2, 0.1] 

original_params = {'learning_rate' : 0.2, 'n_estimators': 250, 'verbose' : True, 'random_state' : 241}
params = dict(original_params)

data = pd.read_csv('gbm-data.csv')
np_data = data.values
X = np_data[:,1:]
y = np_data[:,[0]].ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

clf = ensemble.GradientBoostingClassifier(**params)
clf.fit(X_train, y_train)

test_loss = np.zeros((params['n_estimators'],), dtype=np.float64)
train_loss = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
    # clf.loss_ assumes that y_test[i] in {0, 1}
    y_sig = (1.0 / (1.0 + np.exp(0.0 - y_pred)))
    test_loss[i] = log_loss(y_test, y_sig)#clf.loss_(y_test, y_sig)

for i, y_pred in enumerate(clf.staged_decision_function(X_train)):
    # clf.loss_ assumes that y_test[i] in {0, 1}
    y_sig = (1.0 / (1.0 + np.exp(0.0 - y_pred)))
    train_loss[i] = log_loss(y_train, y_sig)#clf.loss_(y_train, y_sig)

plt.figure()
plt.plot(test_loss, 'r', linewidth=2)
plt.plot(train_loss, 'g', linewidth=2)
plt.legend(['test', 'train'])

i = np.argmin(test_loss)
    
print('min log-loss: ', np.round(test_loss[i],2), ' iteration#: ', i)

rfc = RandomForestClassifier(random_state=241, n_estimators=i)
rfc.fit(X_train, y_train)
y_pred = rfc.predict_proba(X_test)

print('RandomForest log-loss: ', np.round(log_loss(y_test, y_pred),2))