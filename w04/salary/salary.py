import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer, TfidfVectorizer


data_train = pd.read_csv('salary-train.csv')
data_test = pd.read_csv('salary-test-mini.csv')

data_train['FullDescription'] = data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
data_train['FullDescription'] = data_train['FullDescription'].str.lower()
data_test['FullDescription'] = data_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True).str.lower()

tfidf_vectorizer = TfidfVectorizer()
tfidf_tain = tfidf_vectorizer.fit_transform(data_train['FullDescription'])
tfdif_test = tfidf_vectorizer.transform(data_test['FullDescription'])

data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)

enc = DictVectorizer()

X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

#data_train[['LocationNormalized','ContractTime']] = X_train_categ.indices

#X = np.array(content[[1,2]])

print()#, X_train_categ.indices[0])