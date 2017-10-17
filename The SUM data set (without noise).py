# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os

# specify data set path
data_path = os.path.abspath('Data Sets/The SUM dataset/without noise/The SUM dataset, without noise.csv')

# read CSV file directly from path and save the results
data = pd.read_csv(data_path, sep=';', index_col = 0) # 'sep' specifies separator used in the CSV file

# create a python list of feature names
feature_cols = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 6', 'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10']



# use the list to create a subset of the original DataFrame (X)
X = data.loc[:100,feature_cols] # for 100 rows select the 'feature cols'

# alternative version
# X = data.loc[:100,['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 6', 'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10']]


# select the Target column as the response (Y)  
y = data.Target[:100] # select first 100 elements from the Target

# 10-fold cross validation with linear regression, using RMSE (root mean squared error) metric 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

linear_reg = LinearRegression() # instantiate model

scores = cross_val_score(linear_reg, X, y, cv=10, scoring='neg_mean_squared_error')

# note: cross_val_score takes care of splitting the data into folds, 
# so we don't need to split the data ourselves using train/test split

# fix the sign of MSE scores
mse_scores = -scores
print (mse_scores)

# convert from MSE to RMSE
rmse_scores = np.sqrt(mse_scores)

# calculate average RMSE
print (rmse_scores.mean())