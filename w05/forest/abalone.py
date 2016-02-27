import numpy as np
import pandas as pd
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.cross_validation import KFold, cross_val_score
import time


def est(n):
    return RandomForestRegressor(random_state=1, n_estimators=n)

def cross_val_score_n(n):
    t0 = time()
    cvs = cross_val_score(est(n), X, y=y, cv=kf, scoring='r2')
    print("n = %d, cross_val_score done in %fs" % n % (time() - t0))
    return cvs
    
data= pd.read_csv('abalone.csv')
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

y = data['Rings']
X = data[[c for c in data.columns if c not in ['Rings']]]

kf = KFold(len(y), n_folds = 5, shuffle=True, random_state=1)
    
opt = list(map(lambda x: np.mean(cross_val_score_n(x)),range(1,51)))
    
    
"""
pca = PCA(n_components=10)
pca.fit(X)
X_new = pca.transform(X)

djia = pd.read_csv('djia_index.csv')
cor = np.corrcoef(djia['^DJI'],X_new[:,0])

for i in range(len(pca.explained_variance_ratio_)):
    if np.sum(pca.explained_variance_ratio_[0:i]) > 0.9:
        print(i)
        break

cors = list(map(lambda x: np.abs(np.round(np.corrcoef(X_new[:,0],X.ix[:,x])[0][1],2)), range(X.shape[1])))

print(np.round(cor[0][1],2))

#print(X.columns[0])

print(X.columns[np.argmax(cors)])"""

print()