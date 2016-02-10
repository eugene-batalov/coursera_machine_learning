from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.cross_validation import KFold, cross_val_score
import numpy as np


boston = load_boston()
#a = np.asarray(boston.DESCR)
#np.savetxt("boston_data.csv", a, delimiter=",")

data = scale(boston.data)
kf = KFold(data.shape[0], n_folds = 5, shuffle=True, random_state=42)

def knr(power):
    return KNeighborsRegressor(5, weights='distance', p=power, metric='minkowski')

def cross_val_score_p(p):
    return cross_val_score(knr(p), data, y=boston.target, cv=kf, scoring='mean_squared_error')

n = np.linspace(1.0, 10.0, num=200)
r = range(1,11)

opt = list(map(lambda x: np.mean(cross_val_score_p(x)),n))

print("\nmin = ", np.round(np.max(opt),2), "\narg = ", np.argmax(opt), "\np = ", n[np.argmax(opt)])#"\nopt = ", opt, boston.data.shape, boston.target.shape, data.shape)