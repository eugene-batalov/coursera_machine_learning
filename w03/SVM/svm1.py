# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import datasets, grid_search, svm, feature_extraction, cross_validation

newsgroups = datasets.fetch_20newsgroups(subset = 'all', categories = ['alt.atheism', 'sci.space'])
grid = {'C': np.power(10.0, range(-5, 6))}
clf = svm.SVC(random_state = 241, kernel = 'linear')
kf = cross_validation.KFold(len(newsgroups.target), n_folds = 5, shuffle=True, random_state = 241)
vect = feature_extraction.text.TfidfVectorizer()
X = vect.fit_transform(newsgroups.data)
gs_clf = grid_search.GridSearchCV(clf, grid, scoring = 'accuracy', cv = kf)
gs_clf.fit(X, newsgroups.target)
print(gs_clf.best_params_)
#OUT: {'C': 10.0}
c_clf = svm.SVC(C = gs_clf.best_params_['C'], kernel = 'linear', random_state = 241)
ans = c_clf.fit(X, newsgroups.target)
resultInd = np.array(c_clf.coef_.indices)[np.argsort(np.abs(c_clf.coef_.data))[-10:]]
ans = [None]*len(resultInd)
for i in range(len(ans)):
    ans[i] = vect.get_feature_names()[resultInd[i]]
ans.sort()
print(ans)
#OUT ...

"""
word = DataFrame(data=vectorizer.get_feature_names())
coef = DataFrame(data=np.abs(np.asarray(clf.coef_.todense()).reshape(-1)))
data = pd.concat([name, coef], axis=1)
data.columns = ['word', 'coef']
data.sort_index(by=['coef'])[-10:]
"""