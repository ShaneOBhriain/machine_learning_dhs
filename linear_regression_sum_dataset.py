# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import csv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def printResultsToCsv():
    with open("test.csv",'w') as csvfile:
        resultswriter = csv.writer(csvfile, delimiter=";")
        for row in allResults:
            resultswriter.writerow(row)
    return;

def createRegressionModel(type):
    if type == "linear":
        return LinearRegression() # instantiate model
    else:
        print("Incorrect specification of regression model")
        return;

def readData(filename,sep,nrows):
    # read CSV file directly from path and save the results
    data = pd.read_csv(filename, sep=sep, index_col = 0, nrows=nrows)
    # drop null attributes
    data = data.dropna()
    # use the list to create a subset of the original DataFrame (X)
    X = data.loc[:,feature_cols] # for 100 rows select the 'feature cols'
    # select the Target column as the response (Y)
    y = data.Target # select first 100 elements from the Target

    dataDict = {"x": X, "y": y}

    return dataDict;

# data = data dictionary with x and y keys
# regression_model_type: string containing type of regression model
def runRegression(data, regression_model_type):
    print("Running regression")
    # TODO: Dynamic selection of regression model type
    regression_model = createRegressionModel(regression_model_type);

    features = data["x"];
    targets = data["y"];
    # 10-fold cross validation with linear regression, using RMSE (root mean squared error) metric
    kfold = KFold(n_splits=10)
    scores = cross_val_score(regression_model, features, targets, cv=kfold, scoring=regression_metric)
    # note: cross_val_score takes care of splitting the data into folds,
    # so we don't need to split the data ourselves using train/test split
    if regression_metric == "neg_mean_squared_error":
        # fix the sign of MSE scores
        mse_scores = -scores
        # convert from MSE to RMSE
        scores = np.sqrt(mse_scores)

    # calculate average RMSE
    print ("["+ filename+": "+ regression_metric +"] Mean score for sample size " + str(sample_size) + " : " + str(scores.mean()))
    results.append(scores.mean());

    allResults.append(results)

    return;



multiplier = 5
sample_size = 100

answers = {}

# specify data set path
data_path = os.path.abspath('Data Sets/The SUM dataset/without noise/The SUM dataset, without noise.csv')
filename1 = "newsum.csv"
filename2 = "sum_ds_wn.csv"
# filename3 = "sum_ds_nn.csv"
# filename4 = "sum_ds_nn.csv"

filenames = [filename1];
sample_sizes = [100,500,1000]
# sample_sizes = [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]
allResults = []
results = []

regression_metrics = ['neg_mean_squared_error','r2']

# create a python list of feature names
feature_cols = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 6', 'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10']

for filename in filenames:
    for regression_metric in regression_metrics:
        results = [filename+regression_metric]
        for sample_size in sample_sizes:
            # read data
            data = readData(filename,";",sample_size);
            # TODO: For all regression types
            runRegression(data, "linear");



printResultsToCsv()
