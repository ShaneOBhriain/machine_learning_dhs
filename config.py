from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from count_lines import count_lines

# Not measuring effectiveness of algos
using_evaluation = True
evaluation_metrics = ["time","rate_of_improvement"]

models = [  {"name": "Linear Regression","model":linear_model.LinearRegression(), "type": "regression", "enabled":True},
            {"name": "Ridge Regression","model":linear_model.Ridge(), "type": "regression", "enabled":True},
            {"name": "Logistic Regression","model":linear_model.LogisticRegression(), "type": "classification", "enabled":True},
            {"name": "K Neighbours","model":KNeighborsClassifier(20), "type": "classification", "enabled":True}
        ]

regression_metrics = {0: 'neg_mean_squared_error',1: "r2"}
classification_metrics = {0: 'accuracy', 1: make_scorer(precision_score, average="weighted")}

metrics = {"regression": regression_metrics,
           "classification": classification_metrics
          }

def getFileSize (file_info):
    return count_lines(file_info)

def getMaxSampleSize(file_info, model_type):
    max_size = min (getFileSize(file_info), 10000 )
    if model_type == "classification":
        max_size = max(5000, max_size)
    return max_size

### Definition of files
sum_features = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4','Feature 5 (meaningless but please still use it)', 'Feature 6', 'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10']
sum_no_noise_file_info = {
            "name": "newsum.csv",
            "sep":";",
            "regression_target": "Target",
            "classification_target":"Target Class",
            "features": sum_features,
            "needs_transformation": False,
            "too_many_features": False,
            "has_categorical_columns": False,
            "has_labels": True
        }
sum_with_noise_file_info = {
            "name": "sum_ds_wn.csv",
            "sep":";",
            "regression_target": "Noisy Target",
            "classification_target":"Noisy Target Class",
            "features":sum_features,
            "needs_transformation": False,
            "too_many_features": False,
            "has_categorical_columns": False,
            "has_labels": True
        }
big_housing_file_info = {
            "name": "housing_dataset.csv",
            "sep":",",
            "regression_target": "SalePrice",
            "classification_target":"SalePrice",
            "transform_function" : lambda x: round(x/50000),
            "needs_transformation": True,
            "too_many_features": True,
            "drop_features": ["MoSold", "YrSold", "MiscFeature"],
            "has_categorical_columns": True,
            "has_labels": True
        }
kchouse_file_info = {
            "name": "kc_house_data.csv",
            "sep":",",
            "regression_target": "price",
            "classification_target":"price",
            "features": ["bedrooms","bathrooms","sqft_living","sq_loft","grade"],
            "transform_function" : lambda x: round(x/100000),
            "needs_transformation": True,
            "too_many_features": False,
            "has_categorical_columns": False,
            "has_labels": True
        }
wine_file_info = {
            "name": "winequality-red.csv",
            "sep":";",
            "regression_target": "quality",
            "classification_target":"quality",
            "features": ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"],
            "needs_transformation": False,
            "too_many_features": False,
            "has_categorical_columns": False,
            "has_labels": True
        }
taxi_file_info = {
            "name": "taxi.csv",
            "sep":",",
            "regression_target": "trip_duration",
            "classification_target":"trip_duration",
            "features": ["vendor_id","pickup_datetime","dropoff_datetime","passenger_count","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"],
            "transform_function" : lambda x: round(x/60),
            "needs_transformation": True,
            "too_many_features": False,
            "has_categorical_columns": True,
            "has_labels": True
        }
year_prediction_file_info = {
            "name": "YearPredictionMSD.csv",
            "sep":",",
            "target_column_index": 0,
            "features": [],
            "needs_transformation": False,
            "too_many_features": True,
            "has_categorical_columns": False,
            "has_labels": False
        }

### Specify which files you want to run
files =  [sum_no_noise_file_info, sum_with_noise_file_info, taxi_file_info, year_prediction_file_info]

# TODO: team id in file name
result_file_name = "task_results_Team.csv"

# TODO: Multiplier while loop, file sizes
sample_sizes = [50]
# sample_sizes = [50,100,500,1000,5000,10000,50000,100000,500000]
