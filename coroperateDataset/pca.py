from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


ibd_train = pd.read_csv('data/prjeb1220.csv')
X_train = ibd_train.drop(columns=['Disease'])
y_train = ibd_train['Disease']

ibd_test = pd.read_csv('data/r.csv')
X_test = ibd_test.drop(columns=['Disease'])
y_test= ibd_test['Disease']

pca = PCA(n_components=50)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(X_train_pca.clumns)

knn = KNeighborsClassifier()
knn.fit(X_train_pca, y_train)

y_pred = knn.predict(X_test_pca)


score = accuracy_score(y_pred, y_test)
print(score)
