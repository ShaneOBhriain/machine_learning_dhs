# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import csv
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def printResultsToCsv():
    with open("test.csv",'w') as csvfile:
        resultswriter = csv.writer(csvfile, delimiter=";")
        for row in allResults:
            resultswriter.writerow(row)
    return;

def createRegressionModel(type):
    if type == "Linear Regression":
        return linear_model.LinearRegression() # instantiate model
    elif type == "Ridge Regression":
        return linear_model.Ridge()
    else:
        print("Incorrect specification of regression model")
        return;

def readData(filename,nrows):
    # read CSV file directly from path and save the results
    data = pd.read_csv(filename, sep=getSeperator(filename), index_col = 0, nrows=nrows)
    data = data.replace(np.NaN, 0)
    # drop null attributes
    data = data.dropna()

    # use the list to create a subset of the original DataFrame (X)
    X = data.loc[:,getFeatures(filename)]
    # print("X")
    # print(X)
    # select the Target column as the response (Y)
    y = data[getTargetName(filename)]
    dataDict = {"x": X, "y": y}

    return dataDict;

# Creates array with appropriate row name as first element, and scores as tail, for printing to csv
def createResultRow(regression_type,dataset_name, regression_metric_name, scores):
    result_row = [regression_type +"; "+ dataset_name + "; " + regression_metric];
    result_row += scores;
    return result_row;

def addToResults(results_row):
    allResults.append(results_row);
    return;

sum_features = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4','Feature 5 (meaningless but please still use it)', 'Feature 6', 'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10']
housing_features = ["MSSubClass", "LotFrontage", "LotArea", "YearBuilt", "YearRemodAdd", "TotalBsmtSF","1stFlrSF","2ndFlrSF", "GrLivArea","BedroomAbvGr","TotRmsAbvGrd", "Fireplaces"]

features = {"newsum.csv": sum_features, "sum_ds_wn.csv": sum_features, "housing_dataset.csv": housing_features  }

def getFeatures(dataset_name):
    return features[dataset_name]

target_names = {"newsum.csv": "Target", "sum_ds_wn.csv": "Noisy Target", "housing_dataset.csv": "SalePrice"}
def getTargetName(dataset_name):
    return target_names[dataset_name]

def getSeperator(dataset_name):
    if dataset_name == "housing_dataset.csv":
        return ","
    else:
        return ";"

# data = data dictionary with x and y keys
# regression_model_type: string containing type of regression model
def runRegression(data, regression_model_type, regression_metric):
    # TODO: Dynamic selection of regression model type
    lm = createRegressionModel(regression_model_type)
    features = data["x"]
    targets = data["y"]

    # 10-fold cross validation with linear regression, using RMSE (root mean squared error) metric
    kfold = KFold(n_splits=10)
    scores = cross_val_score(lm, features, targets, cv=kfold, scoring=regression_metric)
    # note: cross_val_score takes care of splitting the data into folds,
    # so we don't need to split the data ourselves using train/test split
    if regression_metric == "neg_mean_squared_error":
        # fix the sign of MSE scores
        mse_scores = -scores
        # convert from MSE to RMSE
        scores = np.sqrt(mse_scores)

    # calculate average RMSE
    print ("["+ filename+": "+ regression_metric +"] Error for sample size " + str(sample_size) + " : " + str(scores.mean()))
    return scores.mean();


# specify data set path
# TODO: FIX PATHS, UNZIPPING
data_path = os.path.abspath('Data Sets/The SUM dataset/without noise/The SUM dataset, without noise.csv')
filename1 = "newsum.csv"
filename2 = "sum_ds_wn.csv"
filename3 = "housing_dataset.csv"
# filename4 = "sum_ds_nn.csv"

filenames = [filename1,filename2,filename3];
sample_sizes = [100,500,1000]
# sample_sizes = [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]

first_row = [""] + sample_sizes
allResults = [first_row]
results = []

regression_models = ["Linear Regression", "Ridge Regression"]
regression_metrics = ['neg_mean_squared_error','r2']

# create a python list of feature names

for regression_model in regression_models:
    print(regression_model)
    for filename in filenames:
        for regression_metric in regression_metrics:
            scores = []
            for sample_size in sample_sizes:
                # read data
                data = readData(filename,sample_size);
                # TODO: For all regression types
                score = runRegression(data, regression_model, regression_metric);
                scores.append(score)
            result_row = createResultRow(regression_model, filename,regression_metric, scores);
            addToResults(result_row);
printResultsToCsv()
