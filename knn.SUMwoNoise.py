import numpy as np
import pylab as pl
import pandas as pd
import os
import csv
import random
from sklearn import neighbors, datasets

n_neighbors = 15 #THIS NUMBER IS ARBITRARY

#data path:
file = 'Data Sets/The SUM dataset/without noise/The SUM dataset, without noise.csv'

#import data:
data = pd.read_csv(file, sep=';', index_col = 0) 
data.head() #check out first few rows to make sure it imported ok
feature_cols = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5 (meaningless but please still use it)', 'Feature 6', 'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10']

#X is a design matrix, y is a target vector
X = data.loc[:100,feature_cols] #take first 100 instances
y = data.loc[:100, 'Target Class']

##ADD IN FOR: 30/70 split (if 10-fold cross-val isn't working)
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
##fit model
#knn.fit(X_train,y_train)
##predict response
#pred = knn.predict(X_test)
##evaluate accuracy
#print accuracy_score(y_test, pred)

#get classifier, instantiate
from sklearn.neighbors import KNeighborsClassifier
knn = neighbors.KNeighborsClassifier(n_neighbors)

#10-fold cross validation w/ K Nearest Neighbor Classification, using accuracy metric (k = 15)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
#print(scores)

#find average accuracy score across the 10 folds:
print('MEAN SCORES: ')
print('100')
print(scores.mean())

#now do same for other datasets!

#500
X = data.loc[:500,feature_cols] #take first 100 instances
y = data.loc[:500, 'Target Class']
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print('500')
print(scores.mean())

#1000
X = data.loc[:1000,feature_cols] #take first 100 instances
y = data.loc[:1000, 'Target Class']
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print('1000')
print(scores.mean())

#5000
X = data.loc[:5000,feature_cols] #take first 100 instances
y = data.loc[:5000, 'Target Class']
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print('5000')
print(scores.mean())

#10000
X = data.loc[:10000,feature_cols] #take first 100 instances
y = data.loc[:10000, 'Target Class']
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print('10000')
print(scores.mean())

#50000
X = data.loc[:50000,feature_cols] #take first 100 instances
y = data.loc[:50000, 'Target Class']
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print('50000')
print(scores.mean())

#100000
X = data.loc[:100000,feature_cols] #take first 100 instances
y = data.loc[:100000, 'Target Class']
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print('100000')
print(scores.mean())






##make sure it's working:
#print(data.head())
#print('ran ok')
