from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score

result_file_name = "results.csv"

sample_sizes = [100,500,1000,5000,10000,50000]

sum_features = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4','Feature 5 (meaningless but please still use it)', 'Feature 6', 'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10']

file1 = {
            "name": "newsum.csv",
            "sep":";",
            "regression_target": "Target",
            "classification_target":"Target Class",
            "features": sum_features,
            "needs_transformation": False,
            "too_many_features": False
        }
file2 = {
            "name": "sum_ds_wn.csv",
            "sep":";",
            "regression_target": "Noisy Target",
            "classification_target":"Noisy Target Class",
            "features":sum_features,
            "needs_transformation": False,
            "too_many_features": False
        }
file3 = {
            "name": "housing_dataset.csv",
            "sep":",",
            "regression_target": "SalePrice",
            "classification_target":"SalePrice",
            "transform_function" : lambda x: round(x/50000),
            "needs_transformation": True,
            "too_many_features": True,
            "drop_features": ["MoSold", "YrSold", "MiscFeature"]
        }
file4 = {
            "name": "kc_house_data.csv",
            "sep":",",
            "regression_target": "price",
            "classification_target":"price",
            "features": ["bedrooms","bathrooms","sqft_living","sq_loft","grade"],
            "transform_function" : lambda x: round(x/100000),
            "needs_transformation": True,
            "too_many_features": False
        }
file5 = {
            "name": "winequality-red.csv",
            "sep":";",
            "regression_target": "quality",
            "classification_target":"quality",
            "features": ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"],
            "transform_function" : lambda x: x,
            "needs_transformation": False,
            "too_many_features": False
        }


files =  [file1,file2,file3,file4,file5]

regression_models = ["Linear Regression", "Ridge Regression"]
classification_models = ["Logistic Regression", "K Neighbours"]

models = [ {"name":"Linear Regression", "type": "regression"},
            {"name":"Ridge Regression", "type": "regression"},
            {"name":"Logistic Regression", "type": "classification"},
            {"name":"K Neighbours", "type": "classification"}
        ]


regression_metrics = {0: 'neg_mean_squared_error',1: "r2"}
classification_metrics = {0: 'accuracy', 1: make_scorer(precision_score, average="weighted")}
