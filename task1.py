import pandas as pd
import numpy as np
import os
import csv
import config
import time
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing
from scipy.linalg import norm


######## Results handling for printing to CSV

def getMaxListLength(lists):
    max_len = len(lists[0])
    for l in lists:
        if len(l) > max_len:
            max_len = len(l)
    return max_len
def joinLists(list1,list2):
    joined_list = []
    for i in range(len(list1)):
        joined_list.append(list1[i])
        joined_list += list2
    return joined_list

def getResultColumnLabels():
    if config.using_evaluation:
        return joinLists(config.sample_sizes, config.evaluation_metrics)
    else:
        return config.sample_sizes

first_row = [""] + getResultColumnLabels()
allResults = [first_row]
results = []


temp_result = "results_temp.csv"
def printResultsToCsv(is_final):
    if is_final:
        out_file = config.result_file_name
    else:
        out_file = temp_result
    with open(out_file,'w') as csvfile:
        resultswriter = csv.writer(csvfile, delimiter=";")
        for row in allResults:
            resultswriter.writerow(row)
    return;
# Creates array with appropriate row name as first element, and scores as tail, for printing to csv
def createResultRow(regression_type,dataset_name, regression_metric_name, scores):
    if "str" not in str(type(regression_metric_name)):
        regression_metric_name = "precision"
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

    if sample_size>len(features.index):
        print("Sample size bigger than file")
        return ("N/A","N/A")
    # features = preprocessing.scale(features)
    targets = data["y"]

    # For logisitic regression need to convert labels to values
    # TODO: check why only logisitic regression
    if model_to_run["name"] == "Logistic Regression":
        targets = LabelEncoder().fit_transform(targets.tolist())

    # 10-fold cross validation with linear regression
    kfold = KFold(n_splits=10, random_state=0)
    start_time = time.time()
    scores = cross_val_score(model, features, targets, cv=kfold, scoring=regression_metric)
    end_time = time.time()
    time_taken = end_time - start_time

    scores = fixScores(regression_metric,scores)
    # normalise scores to have all between 0 and 1

    print ("["+ file_info["name"]+": "+ model_to_run["name"] + ":"+ str(regression_metric) +" ] Score for sample size " + str(sample_size) + " : " + str(scores.mean()))
    return (scores.mean(),time_taken, (scores.mean()/time_taken));


# End Creation and Runnning of Regression Model ########
regression_metrics = {0: 'neg_mean_squared_error',1: "r2",2:"neg_mean_absolute_error", 3:"explained_variance", 4: "neg_median_absolute_error" }

def normaliseScores(scores):
    old_max = max(scores)
    old_min = min(scores)
    old_range = old_max - old_min
    new_min = 0
    new_max = 1
    normalised_scores = np.array([(new_min + (((x-old_min)*(new_max-new_min)))/(old_max - old_min)) for x in scores])
    return normalised_scores

def fixScores(regression_metric,scores):
    if "neg" in regression_metric:
        scores = [-x for x in scores]
        if regression_metric == "neg_mean_squared_error":
            return normaliseScores(np.sqrt(scores))
    if max(scores)>1 or min(scores) <0:
        return normaliseScores(scores)
    else:
        return scores

######## Helper methods, transform values/columns for classification
def getFeatures(dataset_name):
    return features[dataset_name]


def getTargetName(file_info, target_type):
    if file_info["has_labels"]:
        return file_info[target_type + "_target"]
    else:
        return file_info["target_column"]

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
        if file_info["has_labels"]:
            x=data
            for feat in file_info["drop_features"]:
                x = x.drop(feat,1)
            x = x.dropna()
            return x
        else:
            return data.drop(data.columns[file_info["target_column_index"]],axis=1)
    else:
        return data.loc[:,file_info["features"]]

######## End helper methods

# reads csv file and returns dictionary {x: features, y: target}
def readData(file_info,nrows,model_type):
    filename= file_info["name"]
    index_col = None
    if file_info["has_labels"]:
        index_col = 0
    data = pd.read_csv(filename, sep=file_info["sep"], index_col = index_col, nrows=nrows)
    data = data.replace(np.NaN, 0)
    data = data.dropna()

    X = getX(data,file_info)
    if file_info["has_labels"]:
        y = data[getTargetName(file_info, model_type)]
    else:
        y = data.ix[:,file_info["target_column_index"]]
    if model_type=="classification" and file_info["needs_transformation"]:
        y = [transformValueToClassValue(i,file_info) for i in (y.tolist())]
        y = pd.Series(data=y)

    dataDict = {"x": X, "y": y}

    return dataDict;

def main():
    for model in config.models:
        if model["enabled"]:
            model_name = model["name"]
            print(model_name)
            for file_info in config.files:
                filename = file_info["name"]
                metrics = config.metrics[model["type"]]
                for i in metrics.keys():
                    metric = metrics[i]
                    scores = []
                    for sample_size in config.sample_sizes:
                        data = readData(file_info,sample_size, model["type"]);
                        (score,time,rate_of_improvement) = runRegression(data, model, metric,file_info,sample_size);
                        scores.append(score)
                        if config.using_evaluation:
                            scores.append(time)
                            scores.append(rate_of_improvement)
                    result_row = createResultRow(model_name, filename,metric, scores);
                    addToResults(result_row);
                    printResultsToCsv(False)
            printResultsToCsv(True)
main()
