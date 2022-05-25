## Problem 4 part 2

import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import NearestNeighbors


## Data Preprossing

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
X_train = df_train.iloc[:, :-1].values
y_train = df_train.iloc[:, 3].values

df_test = df_test.drop(columns=['ID'])
X_test = df_test.iloc[:, :-1].values
y_test = df_test.iloc[:, 3].values

scaler = StandardScaler()
scaler = scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


## Using skLearn classifiers for KNN

classifier = KNeighborsClassifier(n_neighbors=3)
clf = KNeighborsClassifier(n_neighbors=3, metric='euclidean', weights='distance')
classifier.fit(X_train, y_train)
clf.fit(X_train,y_train)



#neighbors=classifier.kneighbors(n_neighbors=3,return_distance=False)
#print(neighbors)


y_pred = classifier.predict(X_test)
y_pred_weighted = clf.predict(X_test)


## predict_proba gives the final probabilities of predictions 
y_pred_prob = classifier.predict_proba(X_test)
y_pred_weighted_prob = clf.predict_proba(X_test)



print("final probabilities of 3-nearest neighbors: ")
print(y_pred_prob)

print("3-nearest neighbors:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("final probabilities of  Euclidean distance weighted 3-nearest neighbors")
print(y_pred_weighted_prob)

print("Euclidean distance weighted 3-nearest neighbors")
print(confusion_matrix(y_test, y_pred_weighted))
print(classification_report(y_test, y_pred_weighted))

#print(help(KNeighborsClassifier))
