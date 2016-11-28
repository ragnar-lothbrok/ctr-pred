import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.datasets import dump_svmlight_file
import sklearn.datasets

# This will load libsvm file
trainData, trainY = sklearn.datasets.load_svmlight_file('/home/raghunandangupta/Downloads/soc_gen_data/cde.txt', multilabel=True, zero_based=False)

columnCount = trainData.shape[1]

df2 = pd.DataFrame(trainY)
df2[columnCount] = df2[0]
print df2.columns


# This will create trainData frame
df = pd.SparseDataFrame([ pd.SparseSeries(trainData[i].toarray().ravel()) 
                              for i in np.arange(trainY.__len__()) ])

# Here we are selecting column which will participate in prediction
X = df.loc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]
y = df2[0]

frames = [df, df2[columnCount]]

df = pd.concat(frames,axis=1)

df[columnCount] = df[0]
df[0] = df2[0]

print df.columns

# Converting again in matrix 
dtrain = xgb.DMatrix(df)

# print dtrain.num_col()
# print dtrain.num_row()

# Below lines are when we dump trainData in file
modifiedFileName = '/home/raghunandangupta/Downloads/soc_gen_data/cde1.txt'
dump_svmlight_file(X, y, modifiedFileName, zero_based=False, multilabel=False)
