import numpy as np
import pandas as pd
import csv, itertools
import statsmodels.formula.api as smf

input_file = "sum_ds_nn.csv"


df = pd.read_csv(input_file, header=0, sep=";", nrows=10)

np.polyfit(df[])




index =0
for key in df.keys():
    print("index " + str(index) + ": " + key)
    index = index+1


# df = df._get_numeric_data()

# # put the numeric column names in a python list
# numeric_headers = list(df.columns.values)
#
# # create a numpy array with the numeric values for input into scikit-learn
# numpy_arracy = df.as_matrix()
#
# # reverse the order of the columns
# numeric_headers.reverse()
# reverse_df = df[numeric_headers]
#
# # write the reverse_df to an excel spreadsheet
# reverse_df.to_csv('reversed_headers.csv')
