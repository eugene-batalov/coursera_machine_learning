import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


data= pd.read_csv('close_prices.csv')
X = data[[c for c in data.columns if c not in ['date']]]
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

print(X.columns[np.argmax(cors)])