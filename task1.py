import pandas as pd
import numpy as np
import os
import csv
import config
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier


def printResultsToCsv(csv_out_name):
    with open(csv_out_name,'w') as csvfile:
        resultswriter = csv.writer(csvfile, delimiter=";")
        for row in allResults:
            resultswriter.writerow(row)
    return;

def createRegressionModel(type):
    if type == "Linear Regression":
        return linear_model.LinearRegression() # instantiate model
    elif type == "Ridge Regression":
        return linear_model.Ridge()
    elif type == "Logistic Regression":
        return linear_model.LogisticRegression() # instantiate
    elif type == "K Neighbours":
        return KNeighborsClassifier(20)
    else:
        print("Incorrect specification of regression model")
        return;


def transformValueToClassValue(value, datafile):
    if "str" in str(type(value)):
        return value
    else:
        return datafile["transform_function"](value)

def transformColumn(column):
    transformed_list = LabelEncoder().fit_transform(column.tolist())
    transformed_series = pd.Series(data=transformed_list)
    transformed_series = transformed_series.replace(np.NaN, 0)
    # Dont know why but the last value is always NaN, maybe an index issue where the list starts at 0 and the series at one, or something
    transformed_series.set_value(len(transformed_series.values),transformed_list[-1])
    return transformed_series

def getX(data, datafile):
    if datafile["name"] == "housing_dataset.csv":
        x = data.drop("MoSold",1)
        x = x.drop("YrSold",1)
        x = x.drop("MiscFeature",1)
        x = x.dropna()
        return x
    else:
        return data.loc[:,datafile["features"]]

def readData(datafile,nrows,is_classification):
    filename= datafile["name"]
    # read CSV file directly from path and save the results
    data = pd.read_csv(filename, sep=datafile["sep"], index_col = 0, nrows=nrows)
    data = data.replace(np.NaN, 0)
    # drop null attributes
    data = data.dropna()

    # use the list to create a subset of the original DataFrame (X)
    X = getX(data,datafile)

    target_column_name = getTargetName(datafile, is_classification)

    y = data[target_column_name]

    if is_classification and "Class" not in target_column_name and "wine" not in filename:
        y = [transformValueToClassValue(i,datafile) for i in (y.tolist())]
        y = pd.Series(data=y)

    dataDict = {"x": X, "y": y}

    return dataDict;


def getFeatures(dataset_name):
    return features[dataset_name]


def getTargetName(datafile, is_classification):
    if is_classification:
        target_name = datafile["classification_target"]
    else:
        target_name = datafile["regression_target"]

    return target_name

def getMetric(model, key, for_label):
    if model == "Linear Regression":
        return config.regression_metrics[key]
    elif model == "Ridge Regression":
        return config.regression_metrics[key]
    else:
        if for_label and key==1:
            return "precision"
        else:
            return config.classification_metrics[key]

# data = data dictionary with x and y keys
# regression_model_type: string containing type of regression model
def runRegression(data, regression_model_type, regression_metric, filename,sample_size):
    lm = createRegressionModel(regression_model_type)
    features = data["x"].replace(np.NaN,0)
    if(filename=="housing_dataset.csv"):
        new_features = []
        for column in features:
            if "object" in str(features[column].dtype):
                features[column] = transformColumn(features[column])

    targets = data["y"]

    # For logisitic regression need to convert labels to values
    if regression_model_type == "Logistic Regression":
        targets = LabelEncoder().fit_transform(targets.tolist())

    # 10-fold cross validation with linear regression, using RMSE (root mean squared error) metric
    kfold = KFold(n_splits=10, random_state=0)
    scores = cross_val_score(lm, features, targets, cv=kfold, scoring=regression_metric)

    if regression_metric == "neg_mean_squared_error":
        # fix the sign of MSE scores
        mse_scores = -scores
        # convert from MSE to RMSE
        scores = np.sqrt(mse_scores)

    print ("["+ filename+": "+ regression_model_type + ":"+ str(regression_metric) +" ] Score for sample size " + str(sample_size) + " : " + str(scores.mean()))
    return scores.mean();

def isClassification(model_name):
    if model_name in config.classification_models:
        return True
    else:
        return False
# Creates array with appropriate row name as first element, and scores as tail, for printing to csv
def createResultRow(regression_type,dataset_name, regression_metric_name, scores):
    result_row = [regression_type +"- "+ dataset_name + "- " + regression_metric_name];
    result_row += scores;
    return result_row;

def addToResults(results_row):
    allResults.append(results_row);
    return;


first_row = [""] + config.sample_sizes
allResults = [first_row]
results = []

all_models = config.regression_models + config.classification_models

def main():
    for model in all_models:
        for datafile in config.files:
            filename = datafile["name"]
            for i in [0,1]:
                metric = getMetric(model, i,False)
                metric_name = getMetric(model, i,True)
                scores = []
                for sample_size in config.sample_sizes:
                    data = readData(datafile,sample_size, isClassification(model));
                    score = runRegression(data, model, metric,filename,sample_size);
                    scores.append(score)
                result_row = createResultRow(model, filename,metric_name, scores);
                addToResults(result_row);
    printResultsToCsv(config.result_file_name)

main()
