from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import grid_search
import numpy as np
from sklearn.cross_validation import KFold


from sklearn import datasets

newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )


y = newsgroups.target
tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(newsgroups.data)


grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(len(newsgroups.data), n_folds=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = grid_search.GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(tfidf, y)

clf0 = SVC(C = 1, kernel='linear', random_state = 241) 
clf0.fit(tfidf,y)
data = clf0.coef_.data
indices = clf0.coef_.indices
a = np.array(clf0.coef_)

names=np.array(tfidf_vectorizer.get_feature_names())

most_often = indices[np.argsort(data)[-10:][::-1]]

print(gs.best_params_,np.sort(names[most_often]))#, len(tfidf_vectorizer.get_feature_names())), gs.grid_scores_