import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.datasets import dump_svmlight_file
import sklearn.datasets

#This will load libsvm file
intialData = sklearn.datasets.load_svmlight_file('/home/raghunandangupta/Downloads/soc_gen_data/cde.txt')

#This will create data frame
df = pd.SparseDataFrame([ pd.SparseSeries(intialData[0][i].toarray().ravel()) 
                              for i in np.arange(2) ])

#Here we are selecting column which will participate in prediction
X = df.loc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24]]
y = df[0]

#Converting again in matrix 
dtrain = xgb.DMatrix(df)

#Below lines are when we dump data in file
# modifiedFileName = '/home/raghunandangupta/Downloads/soc_gen_data/cde1.txt'
# dump_svmlight_file(X, y, modifiedFileName, zero_based=True, multilabel=False)
