import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time
from scipy.sparse import hstack


data_train = pd.read_csv('salary-train.csv')
data_test = pd.read_csv('salary-test-mini.csv')

data_train['FullDescription'] = data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
data_train['FullDescription'] = data_train['FullDescription'].str.lower()
data_test['FullDescription'] = data_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True).str.lower()

t0 = time()
tfidf_vectorizer = TfidfVectorizer(min_df=5)
tfidf_train = tfidf_vectorizer.fit_transform(data_train['FullDescription'])
tfdif_test = tfidf_vectorizer.transform(data_test['FullDescription'])
print("TfidfVectorizer done in %fs" % (time() - t0))

data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)

enc = DictVectorizer()
t0 = time()
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))
print("DictVectorizer done in %fs" % (time() - t0))

X = hstack([tfidf_train, X_train_categ])
y = data_train["SalaryNormalized"]

clf = Ridge(alpha=1.0)
clf.fit(X, y)
#data_train[['LocationNormalized','ContractTime']] = X_train_categ.indices

#X = np.array(content[[1,2]])

print(np.round(clf.predict(hstack([tfdif_test, X_test_categ])),2))#, X_train_categ.indices[0])