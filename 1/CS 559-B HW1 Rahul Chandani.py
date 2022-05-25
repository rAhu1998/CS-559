# Importing Libraries

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split,RepeatedKFold
from sklearn.model_selection import cross_val_score,KFold
from sklearn.linear_model import ElasticNet,Ridge,Lasso
from sklearn.metrics import mean_squared_error

# Reading Data from the Xlsx File into a pandas dataframe

df = pd.read_excel('ENB2012_data.xlsx')

# #Checking for null values 
# print(df.isnull().values.any)

# #Looking for outliers
# z = np.abs(stats.zscore(df))
# #print(z)
# threshold = 3
# #print(np.where(z > 3))

#Splitting the data into input and output features
X = df[["X1","X2","X3","X4","X5","X6","X7","X8"]]
y = df["Y2"]

#Using all the models to train the data
model_Lasso = Lasso(alpha=1.0)

model_Ridge = Ridge(alpha=1.0)

model_ElasticNet = ElasticNet(alpha=1.0, l1_ratio=0.5)

#Using Kfolds = 5 to cross validate the results
kf = KFold(n_splits=5)
kf.get_n_splits(X)

scores_Lasso = []
scores_Ridge = []
scores_ElasticNet= []

mse_Lasso = []
mse_Ridge = []
mse_ElasticNet= []

#Train and test the all three models, calculate their scores and MSE

for train_index, test_index in kf.split(X):
	X_train, X_test = X.iloc[train_index], X.iloc[test_index]
	y_train, y_test = y.iloc[train_index], y.iloc[test_index]
	
	model_Lasso.fit(X_train, y_train)
	model_Ridge.fit(X_train, y_train)
	model_ElasticNet.fit(X_train, y_train)

	scores_Lasso.append(model_Lasso.score(X_test,y_test))
	scores_Ridge.append(model_Ridge.score(X_test,y_test))
	scores_ElasticNet.append(model_ElasticNet.score(X_test,y_test))

	temp = model_Lasso.predict(X_test)
	mse_Lasso.append(mean_squared_error(temp,y_test))
	temp = model_Ridge.predict(X_test)
	mse_Ridge.append(mean_squared_error(temp,y_test))
	temp = model_ElasticNet.predict(X_test)
	mse_ElasticNet.append(mean_squared_error(temp,y_test))


#scores and MSE of 
print("Mean of scores of Lasso:",sum(scores_Lasso)/len(scores_Lasso))
print("Mean of scores of Ridge:",sum(scores_Ridge)/len(scores_Ridge))
print("Mean of scores of ElasticNet:",sum(scores_ElasticNet)/len(scores_ElasticNet))
print("MSE of : \nLasso , Ridge , ElasticNet")
print(sum(mse_Lasso)/len(mse_Lasso),sum(mse_Ridge)/len(mse_Ridge),sum(mse_ElasticNet)/len(mse_ElasticNet))

# #Using 5-fold cross validation
# cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)


# #Traning and Testing 
# scores_Lasso = cross_val_score(model_Lasso, X, y, cv=5)

# scores_Ridge = cross_val_score(model_Ridge, X, y, cv=5)

# scores_ElasticNet = cross_val_score(model_ElasticNet, X, y, cv=5)

# print(scores_Lasso)
# print(scores_Ridge)
# # print(scores_ElasticNet)


# #Mean of all the scores and the standard deviation.
# print("Mean:- "+str(scores_Lasso.mean()) + "  Standard deviation:- "+ str(scores_Lasso.std()))

# print("Mean:- "+str(scores_Ridge.mean()) + "  Standard deviation:- "+ str(scores_Ridge.std()))

# print("Mean:- "+str(scores_ElasticNet.mean()) + "  Standard deviation:- "+ str(scores_ElasticNet.std()))