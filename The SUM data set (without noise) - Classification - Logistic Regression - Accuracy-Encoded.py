# coding: utf-8
import pandas as pd
import numpy as np

# specify data set path
file ='Data Sets/The SUM dataset/without noise/The SUM dataset, without noise.csv'

# read CSV file directly from path and save the results
data = pd.read_csv(file, sep=';', index_col = 0) # 'sep' specifies separator used in the CSV file

# create a python list of feature names
feature_cols = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4','Feature 5 (meaningless but please still use it)', 'Feature 6', 'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10', 'Target']


# 10-fold cross validation with linear regression, using accuracy metric 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


logistic_reg = LogisticRegression() # instantiate 
kfold = KFold(n_splits=10, random_state=0)
scores = cross_val_score(logistic_reg, X, y, cv=kfold, scoring='accuracy')

# Change "Target Class" to numeric values
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
data["target_class_code"] = lb_make.fit_transform(data["Target Class"])


#100 instances
X = data.loc[:100,feature_cols]
y = data.loc[:100, 'target_class_code']

scores = cross_val_score(logistic_reg, X, y, cv=kfold, scoring='accuracy')
print (scores.mean())


#500 instances
X1 = data.loc[:100,feature_cols]
y1 = data.loc[:100, 'target_class_code']

scores = cross_val_score(logistic_reg, X1, y1, cv=kfold, scoring='accuracy')
print (scores.mean())


#1000 instances
X2 = data.loc[:1000,feature_cols]
y2 = data.loc[:1000, 'target_class_code']

scores = cross_val_score(logistic_reg, X2, y2, cv=kfold, scoring='accuracy')
print (scores.mean())

#5000 instances
X3 = data.loc[:5000,feature_cols]
y3 = data.loc[:5000, 'target_class_code']

scores = cross_val_score(logistic_reg, X3, y3, cv=kfold, scoring='accuracy')
print (scores.mean())

#10000 instances
X4 = data.loc[:10000,feature_cols]
y4 = data.loc[:10000, 'target_class_code']

scores = cross_val_score(logistic_reg, X4, y4, cv=kfold, scoring='accuracy')
print (scores.mean())

#50000 instances
X5 = data.loc[:50000,feature_cols]
y5 = data.loc[:50000, 'target_class_code']

scores = cross_val_score(logistic_reg, X5, y5, cv=kfold, scoring='accuracy')
print (scores.mean())

#100000 instances
X6 = data.loc[:100000,feature_cols] 
y6 = data.loc[:100000, 'target_class_code']

scores = cross_val_score(logistic_reg, X6, y6, cv=kfold, scoring='accuracy')
print (scores.mean())


