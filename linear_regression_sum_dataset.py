# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

multiplier = 5
number_of_samples = 100

answers = {}

# specify data set path
data_path = os.path.abspath('Data Sets/The SUM dataset/without noise/The SUM dataset, without noise.csv')
filename = "sum_ds_nn.csv"

# create a python list of feature names
feature_cols = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 6', 'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10']

while number_of_samples <= 100000000:
    # read CSV file directly from path and save the results
    data = pd.read_csv(filename, sep=';', index_col = 0, nrows=number_of_samples) # 'sep' specifies separator used in the CSV file

    # use the list to create a subset of the original DataFrame (X)
    X = data.loc[:,feature_cols] # for 100 rows select the 'feature cols'

    # select the Target column as the response (Y)
    y = data.Target # select first 100 elements from the Target

    # 10-fold cross validation with linear regression, using RMSE (root mean squared error) metric

    linear_reg = LinearRegression() # instantiate model

    scores = cross_val_score(linear_reg, X, y, cv=10, scoring='neg_mean_squared_error')
    # note: cross_val_score takes care of splitting the data into folds,
    # so we don't need to split the data ourselves using train/test split

    # fix the sign of MSE scores
    mse_scores = -scores
    # convert from MSE to RMSE
    rmse_scores = np.sqrt(mse_scores)

    # calculate average RMSE
    print ("Mean score for sample size " + str(number_of_samples) + " : " + str(rmse_scores.mean()))

    number_of_samples = number_of_samples * multiplier
    if multiplier == 5:
        multiplier = 2
    else:
        multiplier = 5
