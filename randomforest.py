#https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/
import pandas as pd
from sklearn.preprocessing.label import LabelEncoder
from sklearn.base import TransformerMixin
import numpy as np
from sklearn.ensemble import RandomForestClassifier

np.random.seed(30)
train_df = pd.read_csv('/Users/raghugupta/Downloads/testdata/train.csv')

test_df = pd.read_csv('/Users/raghugupta/Downloads/testdata/test_1.csv')

exact_test_df = pd.read_csv('/Users/raghugupta/Downloads/testdata/test.csv')


nonnumeric_columns = ['installer', 'basin', 'subvillage', 'scheme_management', 'extraction_type',
                      'management','payment','water_quality','source','waterpoint_type','scheme_management']
feature_columns_to_use = ['id','amount_tsh', 'construction_year', 'installer', 'longitude', 'latitude', 'gps_height', 'basin', 'subvillage', 'region_code',
                          'population', 'scheme_management', 'extraction_type', 'management', 'payment', 'water_quality', 'source', 'waterpoint_type','label']

feature_columns_new = ['id','amount_tsh', 'construction_year', 'installer', 'longitude', 'latitude', 'gps_height', 'basin', 'subvillage', 'region_code',
                          'population', 'scheme_management', 'extraction_type', 'management', 'payment', 'water_quality', 'source', 'waterpoint_type']

for fp in feature_columns_to_use:
    train_df = train_df[pd.notnull(train_df[fp])]
    test_df = test_df[pd.notnull(test_df[fp])]

for fp in feature_columns_new:
    exact_test_df.fillna('NA', inplace=True)
    exact_test_df = exact_test_df[pd.notnull(exact_test_df[fp])]

class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
                              index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


big_train = train_df[feature_columns_to_use]
big_train_imputed = DataFrameImputer().fit_transform(big_train)

big_test = test_df[feature_columns_to_use]
big_test_imputed = DataFrameImputer().fit_transform(big_test)

exact_test = exact_test_df[feature_columns_new]
exact_test_imputed = DataFrameImputer().fit_transform(exact_test)


le = LabelEncoder()
for feature in nonnumeric_columns:
    big_train_imputed[feature] = le.fit_transform(big_train_imputed[feature])
    big_test_imputed[feature] = le.fit_transform(big_test_imputed[feature])
    exact_test_imputed[feature] = le.fit_transform(exact_test_imputed[feature])

train_X = big_train_imputed[0:train_df.shape[0]]
test_X = big_test_imputed[0:test_df.shape[0]]
exact_test_X = exact_test_imputed[0:exact_test_df.shape[0]]

train_X.head(1)

train_y = train_df['label']
test_y = test_df['label']


target = 'label'
idcol = 'id'

rf = RandomForestClassifier(n_estimators=1000, n_jobs=2)
train_X = train_X[feature_columns_new]
rf.fit(train_X, train_y)


print train_X.head(1)
exact_dtest_predictions = rf.predict(exact_test_X)

for num in exact_dtest_predictions:
    if num == 0:
        print "functional";
    elif num == 1:
        print "non functional";
    elif num == 2:
        print "functional needs repair";
    else:
        print "******";
