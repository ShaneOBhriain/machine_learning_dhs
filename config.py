from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score

result_file_name = "results_no_random_state.csv"

sample_sizes = [100,500,1000]
sum_features = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4','Feature 5 (meaningless but please still use it)', 'Feature 6', 'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10']

sum_no_noise_file_info = {
            "name": "newsum.csv",
            "sep":";",
            "regression_target": "Target",
            "classification_target":"Target Class",
            "features": sum_features,
            "needs_transformation": False,
            "too_many_features": False,
            "has_categorical_columns": False
        }
sum_with_noise_file_info = {
            "name": "sum_ds_wn.csv",
            "sep":";",
            "regression_target": "Noisy Target",
            "classification_target":"Noisy Target Class",
            "features":sum_features,
            "needs_transformation": False,
            "too_many_features": False,
            "has_categorical_columns": False
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
            "has_categorical_columns": True
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
            "has_categorical_columns": False
        }
wine_file_info = {
            "name": "winequality-red.csv",
            "sep":";",
            "regression_target": "quality",
            "classification_target":"quality",
            "features": ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"],
            "needs_transformation": False,
            "too_many_features": False,
            "has_categorical_columns": False
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
            "has_categorical_columns": True
        }


files =  [sum_with_noise_file_info]

regression_models = ["Linear Regression", "Ridge Regression"]
classification_models = ["Logistic Regression", "K Neighbours"]

models = [ {"name":"Linear Regression", "type": "regression"},
            {"name":"Ridge Regression", "type": "regression"},
            {"name":"Logistic Regression", "type": "classification"},
            {"name":"K Neighbours", "type": "classification"}
        ]


regression_metrics = {0: 'neg_mean_squared_error',1: "r2"}
classification_metrics = {0: 'accuracy', 1: make_scorer(precision_score, average="weighted")}
