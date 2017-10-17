# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import csv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

def printResultsToCsv():
    with open("test.csv",'w') as csvfile:
        resultswriter = csv.writer(csvfile, delimiter=";")
        for row in allResults:
            resultswriter.writerow(row)
    return;

multiplier = 5
sample_size = 100

answers = {}

# specify data set path
data_path = os.path.abspath('Data Sets/The SUM dataset/without noise/The SUM dataset, without noise.csv')
filename1 = "sum_ds_nn.csv"
filename2 = "sum_ds_wn.csv"
# filename3 = "sum_ds_nn.csv"
# filename4 = "sum_ds_nn.csv"

filenames = [filename1, filename2];
sample_sizes = [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]
allResults = []
results = []
# create a python list of feature names
feature_cols = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 6', 'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10']

for filename in filenames:
    results = [filename]

    for sample_size in sample_sizes:
        # read CSV file directly from path and save the results
        data = pd.read_csv(filename1, sep=';', index_col = 0, nrows=sample_size) # 'sep' specifies separator used in the CSV file

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
        print ("["+ filename +"] Mean score for sample size " + str(sample_size) + " : " + str(rmse_scores.mean()))
        results.append(rmse_scores.mean());

    allResults.append(results)

printResultsToCsv()
