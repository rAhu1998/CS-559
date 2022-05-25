#Problem 4 Part 1


import math
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class Sample():
	def __init__(self, x,y,c=None):
		self.x = x
		self.y = y
		self.c = c 


samples = [Sample(1,1,0),Sample(2,2,0),Sample(1,3,0),Sample(4,1,0),Sample(4,4,0),Sample(5,1,0),Sample(2,6,0),Sample(8,1,0),Sample(9,1,0),Sample(2,8,1),Sample(5,9,1),Sample(6,5,1),Sample(6,8,1),Sample(7,3,1),Sample(7,7,1),Sample(8,7,1),Sample(8,9,1),Sample(9,4,1),Sample(9,6,1)]

test = Sample(5,4)

def manhattanWeightedNeigbours(samples,test,k=3):
	nearestNeighbors = []
	for sample in samples:
		distance = abs(sample.x - test.x) + abs(sample.y - test.y)
		nearestNeighbors.append((distance, sample.c))

	nearestNeighbors = sorted(nearestNeighbors)[:k]

	freqClass1 = 0
	freqClass2 = 0

	for distance in nearestNeighbors:
		if distance[1] == 0:
			freqClass1 += (1/distance[0])
		if distance[1] == 1:
			freqClass2 += (1/distance[0])

	test.c = "-" if freqClass1 > freqClass2 else "+"

	return test


# Using Manhattan Weighted Neigbours 
result = manhattanWeightedNeigbours(samples,test)


X = []
y = []

for sample in samples:
	X.append([sample.x,sample.y])
	y.append(sample.c)

# Calculationg for KNN where n=5
classifier = KNeighborsClassifier(n_neighbors=5)

classifier.fit(X,y)

X_new = np.array([test.x , test.y]).reshape(1,-1)
y_pred = classifier.predict(X_new)

y_pred = "-" if y_pred == 0 else "+"
print("According to 5 nearest neighbors:")
print("The point (5,4) belongs t0 class : " + y_pred)

print("According to Manhattan Weighted Neigbours:")
print("The point (5,4) belongs t0 class : " + str(result.c))

