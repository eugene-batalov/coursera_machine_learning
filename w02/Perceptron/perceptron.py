import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

train_content = pd.read_csv('perceptron-train.csv', header=None)
test_content = pd.read_csv('perceptron-test.csv', header=None)

X_train = train_content.ix[:,1:]
y_train = train_content.ix[:,0]
X_test = test_content.ix[:,1:]
y_test = test_content.ix[:,0]
clf_train = Perceptron(random_state=241)
clf_train.fit(X_train, y_train)
predictions=clf_train.predict(X_test)
acc = accuracy_score(y_test, predictions)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
clf_train = Perceptron(random_state=241)
clf_train.fit(X_train_scaled, y_train)
predictions=clf_train.predict(X_test_scaled)
acc1 = accuracy_score(y_test, predictions)

print(acc, acc1, acc1-acc)

