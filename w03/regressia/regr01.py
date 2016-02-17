import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


content = pd.read_csv('data-logistic.csv', header=None)
X = np.array(content[[1,2]])
Y = np.array(content[[0]])
w1 = 10.0
w2 = 20.0
k = 0.1
C1 = 0.0
C2 = 10.0
 
x1, x2 = X[:,0], X[:,1]
y = Y[:,0]


def f1(w1, w2, k, C, x1, x2, y):
    L = 0
    while L < 1000:
        s1=0
        s2=0
        for i in range(0, len(y)):
           s1 += y[i]*x1[i]*(1.0-(1.0/(1.0+np.exp(-y[i]*(w1*x1[i] + w2*x2[i])))))
           s2 += y[i]*x2[i]*(1.0-(1.0/(1.0+np.exp(-y[i]*(w1*x1[i] + w2*x2[i])))))
           #print(s1,s2)
        wp1 = w1
        w1 += (k*(1.0/len(y))) * s1 - k * C * w1
        wp2 = w2
        w2 += (k*(1.0/len(y))) * s2 - k * C * w2
        delta = float(np.sqrt(np.power((wp1 - w1),2) + np.power((wp2-w2),2)))
        #print(wp1,w1,wp2,w2,delta)
        if(delta < 0.00001):
            break
        L += 1
    a = np.array(1.0/(1.0+np.exp(-w1*x1 - w2*x2)))
    aucroc = roc_auc_score(y, a)
    return L, aucroc

L1, aucroc1 = f1(w1, w2, k, C1, x1, x2, y)
L2, aucroc2 = f1(w1, w2, k, C2, x1, x2, y)

print('L1 = ',L1, ' aucroc1 = ', np.round(aucroc1, 3), ' L2 = ', L2, ' aucroc2 = ', np.round(aucroc2,3))#x1[0], x2[0], len(y))