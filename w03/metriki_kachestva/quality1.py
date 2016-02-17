import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

"""
content = pd.read_csv('classification.csv')

tp = np.isfinite(content['true'][(content['true'] == 1) & (content['pred'] == 1)]).size
fp = np.isfinite(content['true'][(content['pred'] == 1) & (content['true'] == 0)]).size
tn = np.isfinite(content['true'][(content['pred'] == 0) & (content['true'] == 0)]).size
fn = np.isfinite(content['true'][(content['pred'] == 0) & (content['true'] == 1)]).size

y, yp = content['true'], content['pred']

accuracy = accuracy_score(y, yp)
precision = precision_score(y, yp)
recall = recall_score(y,yp)
f1 = f1_score(y, yp)

print(tp,fp,fn,tn)
print(np.round(accuracy, 2),np.round(precision, 2),np.round(recall, 2),np.round(f1, 2))#'tp = ',tp, ' fp = ', fp, 'tn = ',tn, ' fn = ', fn)
"""

scores = pd.read_csv('scores.csv')
from sklearn.metrics import  roc_auc_score, precision_recall_curve


roc_logreg = roc_auc_score(scores['true'], scores['score_logreg'])
roc_smv = roc_auc_score(scores['true'], scores['score_svm'])
roc_knn = roc_auc_score(scores['true'], scores['score_knn'])
roc_tree = roc_auc_score(scores['true'], scores['score_tree'])

def p(scores, sel):
    precision, recall, thresholds = precision_recall_curve(scores['true'], scores[sel])
    
    pmax = precision[0]
    
    for i in range(0,len(precision)):
        if(recall[i] > 0.7):
            if(precision[i] > pmax):
                pmax = precision[i]
    return pmax

pmax1 = p(scores, 'score_logreg')
pmax2 = p(scores, 'score_svm')
pmax3 = p(scores, 'score_knn')
pmax4 = p(scores, 'score_tree')


print(pmax1,pmax2,pmax3,pmax4,'\n', roc_logreg, roc_smv, roc_knn, roc_tree)

