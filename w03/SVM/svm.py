import pandas as pd
import sklearn.cross_validation as cv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
import numpy as np

#def neigh(x):
#    KNeighborsClassifier(n_neighbors=3)

content = [x[3:].strip(' \n') for x in open('wine_attrs.txt')]
content = pd.read_csv('wine.data', names=content)

X = scale(content.ix[:,1:])
#print(X)
#X1 = scale(X)
#print(X1)
y = content.ix[:,0]

kf = cv.KFold(content.shape[0], n_folds = 5, shuffle=True, random_state=42)
#for train_indices, test_indices in kf:
 #   print('Train: %s | test: %s' % (train_indices, test_indices))

neigh = KNeighborsClassifier(n_neighbors=29)
acc = list(map(lambda x: np.mean(cv.cross_val_score(KNeighborsClassifier(n_neighbors=x), X, y=y, cv=kf)), range(1,51)))

acc29 = [neigh.fit(X[train], y.ix[train]).score(X[test], y.ix[test]) for train, test in kf]

print(acc, np.round(np.max(acc),2), np.argmax(acc), np.mean(acc29))#, np.mean(acc3))