import pandas as pd
import numpy as np
import os
import csv
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import precision_score
from sklearn.metrics import make_scorer
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

def createHouseTargetClass(value):
    # print ("CREATING HOUSE TARGET: FROM: " + str(value) + " TO: " + str(round(value / 50000.0)))
    return round(value / 50000.0)

def createKcTargetClass(value):
    return round(value/100000)

transform_functions = {"housing_dataset.csv": createHouseTargetClass, "kc_house_data.csv": createKcTargetClass }

def transformValueToClassValue(value, filename):
    if "str" in str(type(value)):
        return value
    else:
        return transform_functions[filename](value)

def transformColumn(column):
    transformed_list = LabelEncoder().fit_transform(column.tolist())
    transformed_series = pd.Series(data=transformed_list)
    transformed_series.replace(np.NaN, 0)
    transformed_series.set_value(100,2)
    return transformed_series

def getX(data, filename):
    if filename == "housing_dataset.csv":
        x = data.drop("MoSold",1)
        x = x.drop("YrSold",1)
        x = x.drop("MiscFeature",1)
        x = x.dropna()
        return x
    else:
        return data.loc[:,getFeatures(filename)]

def readData(filename,nrows,is_classification):
    # read CSV file directly from path and save the results
    data = pd.read_csv(filename, sep=getSeperator(filename), index_col = 0, nrows=nrows)
    data = data.replace(np.NaN, 0)
    # drop null attributes
    data = data.dropna()

    # use the list to create a subset of the original DataFrame (X)
    X = getX(data,filename)

    target_column_name = getTargetName(filename, is_classification)

    y = data[target_column_name]

    if is_classification and "Class" not in target_column_name:
        y = [transformValueToClassValue(i,filename) for i in (y.tolist())]
        y = pd.Series(data=y)

    dataDict = {"x": X, "y": y}

    return dataDict;



sum_features = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4','Feature 5 (meaningless but please still use it)', 'Feature 6', 'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10']
# TODO: Fix housing_features
housing_features = ["MSSubClass", "LotFrontage", "LotArea", "YearBuilt", "YearRemodAdd", "TotalBsmtSF","1stFlrSF","2ndFlrSF", "GrLivArea","BedroomAbvGr","TotRmsAbvGrd", "Fireplaces"]
kc_house_features = ["bedrooms","bathrooms","sqft_living","sq_loft","grade"]
features = {"newsum.csv": sum_features, "sum_ds_wn.csv": sum_features, "housing_dataset.csv": housing_features, "kc_house_data.csv": kc_house_features  }

def getFeatures(dataset_name):
    return features[dataset_name]

target_names_regression = {"newsum.csv": "Target", "sum_ds_wn.csv": "Noisy Target", "housing_dataset.csv": "SalePrice","kc_house_data.csv":"price"}
target_names_classification = {"newsum.csv": "Target Class", "sum_ds_wn.csv": "Noisy Target Class", "housing_dataset.csv": "SalePrice", "kc_house_data.csv":"price"}
def getTargetName(dataset_name, is_classification):
    if is_classification:
        target_name = target_names_classification[dataset_name]
    else:
        target_name = target_names_regression[dataset_name]

    return target_name

def getSeperator(dataset_name):
    if "hous" in dataset_name in dataset_name:
        return ","
    else:
        return ";"

regression_metrics = {0: 'neg_mean_squared_error',1: "r2"}
classification_metrics = {0: 'accuracy', 1: make_scorer(precision_score, average="weighted")}

def getMetric(model, key, for_label):
    if model == "Linear Regression":
        return regression_metrics[key]
    elif model == "Ridge Regression":
        return regression_metrics[key]
    else:
        if for_label and key==1:
            return "precision"
        else:
            return classification_metrics[key]

# data = data dictionary with x and y keys
# regression_model_type: string containing type of regression model

def runRegression(data, regression_model_type, regression_metric, filename,sample_size):
    # TODO: Dynamic selection of regression model type
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

    print ("["+ filename+": "+ regression_model_type + ":"+ str(regression_metric) +" ] Error for sample size " + str(sample_size) + " : " + str(scores.mean()))
    return scores.mean();

def isClassification(model_name):
    if model_name in classification_models:
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

# specify data set path
# TODO: FIX PATHS, UNZIPPING
data_path = os.path.abspath('Data Sets/The SUM dataset/without noise/The SUM dataset, without noise.csv')
filename1 = "newsum.csv"
filename2 = "sum_ds_wn.csv"
filename3 = "housing_dataset.csv"
filename4 = "kc_house_data.csv"

sample_sizes = [100]
filenames = [filename1,filename2,filename3,filename4];
# sample_sizes = [100,500,1000,5000,10000,50000,100000,500000,1000000,5000000,10000000,50000000,100000000]

first_row = [""] + sample_sizes
allResults = [first_row]
results = []

regression_models = ["Linear Regression", "Ridge Regression"]
classification_models = ["Logistic Regression", "K Neighbours"]

all_models = regression_models + classification_models

def main():
    for model in all_models:
        print(model)
        for filename in filenames:
            print(filename)
            for i in [0,1]:
                metric = getMetric(model, i,False)
                metric_name = getMetric(model, i,True)
                scores = []
                for sample_size in sample_sizes:
                    data = readData(filename,sample_size, isClassification(model));
                    score = runRegression(data, model, metric,filename,sample_size);
                    scores.append(score)

                result_row = createResultRow(model, filename,metric_name, scores);
                addToResults(result_row);
    printResultsToCsv("results.csv")

main()
