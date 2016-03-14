import pandas
import time
import datetime
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV


filename = './features.csv'

#load data
features = pandas.read_csv(filename, index_col='match_id')

#reset index because KFold will use indexes which are not present in real data
features.reset_index(drop=True,inplace=True)

# extract target values
y = features['radiant_win'] 

# remove future predict features
X = features.drop(['duration', 'radiant_win', 'tower_status_radiant', 'tower_status_dire', 'barracks_status_radiant', 'barracks_status_dire'], axis=1)

#search for missed features values
counts = X.count()
print("Columns withs missed feature vlaues:")
print(counts[counts != X.shape[0]])

#fill missed values
X.fillna(0.,inplace = True);

#fix random generator seed
randState = 241

#make cross validation data indices
kf = KFold(X.shape[0], n_folds=5, shuffle = True, random_state=randState)

#manual cross validation, can be replaced with GridSearchCV
def scoreClf(clf, kf):
    #print ('Start training, ...')
    start_time = datetime.datetime.now()

    scores = []
    for train, test in kf:
        clf.fit(X[train,:],y[train])

        y_pred = clf.predict_proba(X[test,:])[:, 1]

        score = roc_auc_score(y[test], y_pred)
        scores.append(score)
        #print ("Score : ", score)

    timeTotal = datetime.datetime.now() - start_time
    avgScroe = np.mean(scores)
    print ('Avg. score = {0} Time elapsed = {1}'.format(avgScroe, timeTotal))
    return avgScroe

def boostingTest():
    print ('GB testing')
    learningRate = 0.1 #can speed up
    treeDepth = 3 #can speed up
    treesCounts = [5, 10, 20 ,30, 40, 50]

    bestEstimator = [0,0]
    for treesCount in treesCounts:
        print ('Trees count = {} ...'.format(treesCount))
        
        clf = GradientBoostingClassifier(learning_rate = learningRate,
                                         n_estimators = treesCount,
                                         max_depth = treeDepth,
                                         random_state = randState)
        s = scoreClf(clf, kf)
        if s > bestEstimator[0]:
            bestEstimator[0] = s
            bestEstimator[1] = treesCount

    print ('Best GB estimator : Trees = {0}, score = {1}'.format(bestEstimator[1],bestEstimator[0]))

def regressionTest():
    print ('LR testing')
    C_vals = np.power(10.0, np.arange(-5, 6))

    bestEstimator = [0,0,0]
    for c in C_vals:
        print ('LR, C = {} ...'.format(c))
        
        clf = LogisticRegression(penalty='l2',
                                 C=c,
                                 random_state = randState)

        s = scoreClf(clf, kf)
        if s > bestEstimator[0]:
            bestEstimator[0] = s
            bestEstimator[1] = c
            bestEstimator[2] = clf
            
    print ('Best LR estimator : С = {0}, score = {1}'.format(bestEstimator[1],bestEstimator[0]))
    return bestEstimator[2]        

#save original data
X_orig = X.copy(True)

#Gradient boosting
X = X_orig.values
boostingTest()

scaler = StandardScaler() #only needed for LR

#LR with category features
print ('LR with category features')
X = scaler.fit_transform(X_orig)
regressionTest()



#LR without category features
print ('LR without category features')
colHeroesNames=[col for col in X_orig.columns if 'hero' in col]
colNames = ['lobby_type']
X = X_orig.copy(True)
X.drop(colHeroesNames, axis=1,inplace=True)
X.drop(colNames, axis=1,inplace=True)
X = scaler.fit_transform(X)
regressionTest()


#LR with transformed category features
print ('LR with transformed category features')

#count unique heroes
uniqueHeroes = np.unique(X_orig[colHeroesNames]).ravel()

N = uniqueHeroes.size

print('Unique heroes count = {}'.format(N))

X_hero = np.zeros((X.shape[0], N))

heroIndex = dict()

i = 0
for h in uniqueHeroes:
    if not(h in heroIndex):
        heroIndex[h] = i
        i += 1

for i in range(X.shape[0]):
    for p in range(5):
        X_hero[i, heroIndex[X_orig.ix[i, 'r%d_hero' % (p+1)]]] = 1
        X_hero[i, heroIndex[X_orig.ix[i, 'd%d_hero' % (p+1)]]] = -1

X = np.concatenate([X, X_hero], axis=1)
bestClf = regressionTest()

#test final model

filename = './features_test.csv'
features = pandas.read_csv(filename, index_col='match_id')
features.fillna(0.,inplace = True);

X = features.drop(colNames, axis=1)
X = X.drop(colHeroesNames, axis=1)
X = scaler.fit_transform(X)

X_hero = np.zeros((X.shape[0], N))

for i in range(X.shape[0]):
    for p in range(5):
        X_hero[i, heroIndex[X_orig.ix[i, 'r%d_hero' % (p+1)]]] = 1
        X_hero[i, heroIndex[X_orig.ix[i, 'd%d_hero' % (p+1)]]] = -1

X = np.concatenate([X, X_hero], axis=1)

y_test_pred = bestClf.predict_proba(X)[:, 1]
            
print('Max prob = {0} Min prob = {1}'.format(np.max(y_test_pred),np.min(y_test_pred)))

df = pandas.DataFrame({'match_id':features.index.values,
                       'radiant_win':y_test_pred})
df =  df.drop('match_id',axis=1)

df.to_csv('./test_predict.csv')

