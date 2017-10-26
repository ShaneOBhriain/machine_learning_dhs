import pandas as pd
import numpy as np
import os
import csv
import config
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from scipy.linalg import norm

######## Results handling for printing to CSV

first_row = [""] + config.sample_sizes
allResults = [first_row]
results = []

def printResultsToCsv():
    with open(config.result_file_name,'w') as csvfile:
        resultswriter = csv.writer(csvfile, delimiter=";")
        for row in allResults:
            resultswriter.writerow(row)
    return;
# Creates array with appropriate row name as first element, and scores as tail, for printing to csv
def createResultRow(regression_type,dataset_name, regression_metric_name, scores):
    result_row = [regression_type +"- "+ dataset_name + "- " + regression_metric_name];
    result_row += scores;
    return result_row;

def addToResults(results_row):
    allResults.append(results_row);
    return;

# End Results Handling ########

######## Creation and Running of Regression Model

# data = dictionary, x and y keys containing X and Y or features and targets
# model_to_run: object, contains name, mdoel and type of regression model
# regression_metric = string, scoring metric for cross_val score
def runRegression(data, model_to_run, regression_metric, file_info,sample_size):
    model = model_to_run["model"]
    # replace NaN values with zero, since dropna() leads to inconsistent input error
    features = data["x"].replace(np.NaN,0)
    if(file_info["has_categorical_columns"]):
        for column in features:
            if "object" in str(features[column].dtype):
                features[column] = transformColumn(features[column]).values

    targets = data["y"]

    # For logisitic regression need to convert labels to values
    # TODO: check why only logisitic regression
    if model_to_run["name"] == "Logistic Regression":
        targets = LabelEncoder().fit_transform(targets.tolist())

    # 10-fold cross validation with linear regression
    kfold = KFold(n_splits=10, random_state=0)
    scores = cross_val_score(model, features, targets, cv=kfold, scoring=regression_metric)

    # Convert from MSE to RMSE scores
    if regression_metric == "neg_mean_squared_error":
        mse_scores = -scores
        scores = np.sqrt(mse_scores)
    # normalise scores to have all between 0 and 1
        scores_norm = norm(scores)
        scores = np.array([x/scores_norm for x in scores])
    print ("["+ file_info["name"]+": "+ model_to_run["name"] + ":"+ str(regression_metric) +" ] Score for sample size " + str(sample_size) + " : " + str(scores.mean()))
    return scores.mean();

# End Creation and Runnning of Regression Model ########

######## Helper methods, transform values/columns for classification
def getFeatures(dataset_name):
    return features[dataset_name]


def getTargetName(file_info, target_type):
    return file_info[target_type + "_target"]

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

def transformValueToClassValue(value, file_info):
    if "str" in str(type(value)):
        return value
    else:
        return file_info["transform_function"](value)

def transformColumn(column):
    transformed_list = LabelEncoder().fit_transform(column.tolist())
    transformed_series = pd.Series(data=transformed_list)
    transformed_series = transformed_series.replace(np.NaN, 0)
    # Dont know why but the last value is always NaN, maybe an index issue where the list starts at 0 and the series at one, or something
    # transformed_series.set_value(len(transformed_series.values),transformed_list[-1])
    return transformed_series


def getX(data, file_info):
    if file_info["too_many_features"]:
        x=data
        for feat in file_info["drop_features"]:
            x = x.drop(feat,1)
        x = x.dropna()
        return x
    else:
        return data.loc[:,file_info["features"]]

######## End helper methods

# reads csv file and returns dictionary {x: features, y: target}
def readData(file_info,nrows,model_type):
    filename= file_info["name"]
    data = pd.read_csv(filename, sep=file_info["sep"], index_col = 0, nrows=nrows)
    data = data.replace(np.NaN, 0)
    data = data.dropna()

    X = getX(data,file_info)
    y = data[getTargetName(file_info, model_type)]

    if model_type=="classification" and file_info["needs_transformation"]:
        y = [transformValueToClassValue(i,file_info) for i in (y.tolist())]
        y = pd.Series(data=y)

    dataDict = {"x": X, "y": y}

    return dataDict;

def main():
    for model in config.models:
        model_name = model["name"]
        for file_info in config.files:
            filename = file_info["name"]
            for i in [0,1]:
                metric = getMetric(model_name, i,False)
                metric_name = getMetric(model_name, i,True)
                scores = []
                for sample_size in config.sample_sizes:
                    data = readData(file_info,sample_size, model["type"]);
                    score = runRegression(data, model, metric,file_info,sample_size);
                    scores.append(score)
                result_row = createResultRow(model_name, filename,metric_name, scores);
                addToResults(result_row);
    printResultsToCsv()

main()
