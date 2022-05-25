'''

Problem 2. Gradient Descent Algorithm and Logistic Regression (40 points)


(2)  Please download the breast cancer data set from UCI Machine Learning repository. Implement your 
Logistic  regression  classifier  with  ML  estimator  using  Stochastic  gradient  descent  and  Mini-Batch 
gradient  descent  algorithms.  Do  not  use  any  package/tool.  Use  cross-validation  for  evaluation  and 
report  the  recall,  precision,  and  accuracy  on  malignant  class  prediction  (class  label  malignant  is 
positive). Write down each step of your solution. 
Link to the data: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29  

'''

## Importing dependencies 


import pandas as pd 
import numpy as np


### Applying Logistic Regression

class LogisticRegression:

	def __init__(self, lr=0.001, itirations=10000):
		self.lr = lr
		self.itirations = itirations
		self.w = None
		self.b = None

	def fit(self, X, y):
		samples , features = X.shape
		self.w = np.zeros(features)
		self.w = self.w.astype(float)
		self.b = 0 

		# gradient desent 
		for _ in range(self.itirations):
			## Simple Linear Regression
			linearRegression = np.dot(X, self.w) + self.b
			
			##Calling sigmoid Function(Logistic Regression)
			perdicted_y = self._sigmoid(linearRegression)	

			# Gradient Desent 
			dw = (1 / samples) * (np.dot(X.T , (perdicted_y - y))) 
			db = (1 / samples) * (np.sum(perdicted_y - y))

			self.w -= self.lr * dw
			self.b -= self.lr * db

##		Mini Batch Gradient Desent
## 		Use Batch Size = 1 for stocastic Gradient Desent
	def fit_minibatch(self,X,y,batchsize = 100):
		batches_X = X.shape[0] // batchsize
		batches_y = y.shape[0] // batchsize
		mini_batches_X = []
		mini_batches_y = []

		for i in np.array_split(X, batches_X):
			mini_batches_X.append(i)
		for i in np.array_split(y, batches_y):
			mini_batches_y.append(i)

		for i, j in zip(mini_batches_X,mini_batches_y):
			self.fit(i,j)
 

	# Prediction Function
	def perdict(self, X):
		linearRegression = np.dot(X, self.w) + self.b
		perdicted_y = self._sigmoid(linearRegression)

		perdicted_y_classes = [2 if i>0.5 else 4 for i in perdicted_y]
		return perdicted_y_classes 


	###	Sigmoid Function
	def _sigmoid(self,X):
		return 1 / (1 + np.exp(-X))



## Breast Cancer Wisconsin Dataset

##Link to the data: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29  

df = pd.read_csv("breast-cancer-wisconsin.data", names = ["id",1,2,3,4,5,6,7,8,9,"B(2) or M(4)"])


#### REMOVE OUTLYERS AND GARBAGE VALUES ####

df = df[(df != '?')]
df.replace('?',-99999, inplace=True)
numeric_cols = df.select_dtypes(exclude='number')
df.drop(numeric_cols, axis=0, inplace=True)

df = df.dropna(subset=[6])
df[6] = df[6].astype('int')



### Seperating data into Test and Train 
test_size = 0.15
train_data = df[:-int(test_size*len(df))]
test_data = df[-int(test_size*len(df)):]

X_train = train_data.iloc[:, 1:-1]
y_train = train_data.iloc[:, -1]

X_test = test_data.iloc[:, 1:-1]
y_test = test_data.iloc[:, -1]


### To mesure Accuracy of the model


def acc(y, y_predicted):
	acc = np.sum(y == y_predicted) / len(y)
	return acc

model = LogisticRegression(lr = 0.0001 ,itirations = 10000)


###Traning The data on Breast Cancer Wiscosin DataSet
model.fit_minibatch(X_train,y_train)
predicted_values = model.perdict(X_test)

print("The accuracy of the model is: ")
print(acc(y_test,predicted_values))

##### Current Acc 0.794
