from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score

result_file_name = "results.csv"

sample_sizes = [1000]

filename1 = "newsum.csv"
filename2 = "sum_ds_wn.csv"
filename3 = "housing_dataset.csv"
filename4 = "kc_house_data.csv"
filename5 = "winequality-red.csv"

filenames = [filename1,filename2,filename3,filename4, filename5];

target_names_regression = {"newsum.csv": "Target", "sum_ds_wn.csv": "Noisy Target", "housing_dataset.csv": "SalePrice","kc_house_data.csv":"price", "winequality-red.csv": "quality"}
target_names_classification = {"newsum.csv": "Target Class", "sum_ds_wn.csv": "Noisy Target Class", "housing_dataset.csv": "SalePrice", "kc_house_data.csv":"price", "winequality-red.csv": "quality"}

sum_features = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4','Feature 5 (meaningless but please still use it)', 'Feature 6', 'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10']

file1 = {
            "name": filename1,
            "sep":";",
            "regression_target": "Target",
            "classification_target":"Target Class",
            "features": sum_features
        }
file2 = {
            "name": filename2,
            "sep":";",
            "regression_target": "Noisy Target",
            "classification_target":"Noisy Target Class",
            "features":sum_features
        }
file3 = {
        "name": filename3,
        "sep":",",
            "regression_target": "SalePrice",
            "classification_target":"SalePrice",
            "features": ["MSSubClass", "LotFrontage", "LotArea", "YearBuilt", "YearRemodAdd", "TotalBsmtSF","1stFlrSF","2ndFlrSF", "GrLivArea","BedroomAbvGr","TotRmsAbvGrd", "Fireplaces"],
            "transform_function" : lambda x: round(x/50000)
        }
file4 = {
        "name": filename4,
        "sep":",",
            "regression_target": "price",
            "classification_target":"price",
            "features": ["bedrooms","bathrooms","sqft_living","sq_loft","grade"],
            "transform_function" : lambda x: round(x/100000)
        }
file5 = {
        "name": filename5,
        "sep":";",
            "regression_target": "quality",
            "classification_target":"quality",
            "features": ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"],
            "transform_function" : lambda x: x
        }


files =  [file1,file2,file3,file4,file5]

regression_models = ["Linear Regression", "Ridge Regression"]
classification_models = ["Logistic Regression", "K Neighbours"]

regression_metrics = {0: 'neg_mean_squared_error',1: "r2"}
classification_metrics = {0: 'accuracy', 1: make_scorer(precision_score, average="weighted")}
