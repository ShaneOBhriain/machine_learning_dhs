# coding: utf-8
import pandas as pd
import numpy as np

# specify data set path
file ='Data Sets/The SUM dataset/without noise/The SUM dataset, without noise.csv'

# read CSV file directly from path and save the results
data = pd.read_csv(file, sep=';', index_col = 0) # 'sep' specifies separator used in the CSV file

# create a python list of feature names
feature_cols = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4','Feature 5 (meaningless but please still use it)', 'Feature 6', 'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10', 'Target']


X = data.loc[:100, feature_cols] 



# select the Target column as the response (Y)  
y = data.loc[:100,'Target Class'] # select first 100 elements from the Target


# 10-fold cross validation with linear regression, using accuracy metric 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


logistic_reg = LogisticRegression() # instantiate 
kfold = KFold(n_splits=10, random_state=0)
scores = cross_val_score(logistic_reg, X, y, cv=kfold, scoring='accuracy')

# note: cross_val_score takes care of splitting the data into folds, 
# so we don't need to split the data ourselves using train/test split

# print average of accuracy scores
print (scores.mean())


#500 instances
X1 = data.loc[:500,feature_cols]

y1 = data.loc[:500, 'Target Class']


scores = cross_val_score(logistic_reg, X1, y1, cv=10, scoring='accuracy')

# calculate average accuracy
print (scores.mean())

#1000 instances
X2 = data.loc[:1000,feature_cols]

y2 = data.loc[:1000, 'Target Class']


scores = cross_val_score(logistic_reg, X2, y2, cv=kfold, scoring='accuracy')

# calculate average accuracy
print (scores.mean())

#5000 instances
X3 = data.loc[:5000,feature_cols]

y3 = data.loc[:5000, 'Target Class']


scores = cross_val_score(logistic_reg, X3, y3, cv=kfold, scoring='accuracy')

# calculate average accuracy
print (scores.mean())

#10000 instances
X4 = data.loc[:10000,feature_cols]

y4 = data.loc[:10000, 'Target Class']


scores = cross_val_score(logistic_reg, X4, y4, cv=kfold, scoring='accuracy')

# calculate average accuracy
print (scores.mean())

#50000 instances
X5 = data.loc[:50000,feature_cols]

y5 = data.loc[:50000, 'Target Class']


scores = cross_val_score(logistic_reg, X5, y5, cv=kfold, scoring='accuracy')

# calculate average accuracy
print (scores.mean())

#100000 instances
X6 = data.loc[:100000,feature_cols] 

y6 = data.loc[:100000, 'Target Class']


scores = cross_val_score(logistic_reg, X6, y6, cv=kfold, scoring='accuracy')

# calculate average accuracy
print (scores.mean())


