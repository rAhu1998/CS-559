'''
Problem 1. Linear Discriminant Analysis (20 points)  
Please download the Iris data set from the UCI Machine Learning repository and implement Linear 
Discriminant Analysis for each pair of the classes and report your results. Note that there are three (3) 
class labels in this data set. To implement these models, you can use python and the sklearn packages. 
Please submit the code along with each step of your solutions to get full points. 
Link to the data: https://archive.ics.uci.edu/ml/datasets/Iris
'''


## Importing dependencies  

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

## Importing data from the iris data set

df = pd.read_csv("iris.data", names = [1,2,3,4,'names'])

## Splitting the data into test and train data frames for  Linear Discriminant Analysis

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

## LDA for all the classes at once

lda = LinearDiscriminantAnalysis()
X_r = lda.fit(X,y).transform(X)
print(lda.explained_variance_ratio_)

## Splitting into 3 pairs fo diffrent classes to perfrom LDA seperatly 

df1 = df[df['names']!='Iris-setosa']
df2 = df[df['names']!='Iris-versicolor']
df3 = df[df['names']!='Iris-virginica']

X1 = df1.iloc[:, :-1]
y1 = df1.iloc[:, -1]

X2 = df2.iloc[:, :-1]
y2 = df2.iloc[:, -1]

X3 = df3.iloc[:, :-1]
y3 = df3.iloc[:, -1]


## Performing LDA on each pair diffrently 

#Using the two classes Iris-versicolor and Iris-virginica 
lda1 = LinearDiscriminantAnalysis()
X1_r2 = lda1.fit(X1,y1).transform(X1)
#print(lda1.explained_variance_ratio_)

#Using the two classes Iris-setosa and Iris-virginica
lda2 = LinearDiscriminantAnalysis()
X2_r2 = lda2.fit(X2,y2).transform(X2)
#print(lda2.explained_variance_ratio_)

#Using the two classes Iris-setosa and Iris-versicolor
lda3 = LinearDiscriminantAnalysis()
X3_r2 = lda3.fit(X3,y3).transform(X3)
#print(lda3.explained_variance_ratio_)

## Plots to show the results
count=0
for i in X1_r2:
	if count < 50:
		plt.scatter(i,0,c="g")
		count = count+1
	else:
		plt.scatter(i,0,c="r")
#plt.scatter(X1_r2[:,0],np.zeros(len(X1_r2)))
plt.show()

count=0
for i in X2_r2:
	if count < 50:
		plt.scatter(i,0,c="g")
		count = count+1
	else:
		plt.scatter(i,0,c="r")
plt.show()

count=0
for i in X3_r2:
	if count < 50:
		plt.scatter(i,0,c="g")
		count = count+1
	else:
		plt.scatter(i,0,c="r")
plt.show()


########### Results are attached with the submission ##################