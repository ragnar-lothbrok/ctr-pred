#https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/
import xgboost as xgb
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.base import TransformerMixin
from sklearn.preprocessing.label import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import  pickle


train_df = pd.read_csv('/Users/raghugupta/Downloads/testdata/train.csv')
# print train_df.head(2)
# print train_df.info()

test_df = pd.read_csv('/Users/raghugupta/Downloads/testdata/test_1.csv')

exact_test_df = pd.read_csv('/Users/raghugupta/Downloads/testdata/test.csv')


nonnumeric_columns = ['date_recorded', 'installer', 'basin', 'subvillage', 'region', 'scheme_management', 'extraction_type',
                      'management','payment','water_quality','source','waterpoint_type','scheme_management']
feature_columns_to_use = ['id','amount_tsh', 'date_recorded', 'installer', 'longitude', 'latitude', 'gps_height', 'basin', 'subvillage', 'region',
                          'population', 'scheme_management', 'extraction_type', 'management', 'payment', 'water_quality', 'source', 'waterpoint_type','label']

feature_columns_new = ['id','amount_tsh', 'date_recorded', 'installer', 'longitude', 'latitude', 'gps_height', 'basin', 'subvillage', 'region',
                          'population', 'scheme_management', 'extraction_type', 'management', 'payment', 'water_quality', 'source', 'waterpoint_type']

for fp in feature_columns_to_use:
    train_df = train_df[pd.notnull(train_df[fp])]
    test_df = test_df[pd.notnull(test_df[fp])]

print exact_test_df.shape
for fp in feature_columns_new:
    exact_test_df.fillna('NA', inplace=True)
    exact_test_df = exact_test_df[pd.notnull(exact_test_df[fp])]
print exact_test_df.shape


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

train_y = train_df['label']
test_y = test_df['label']


target = 'label'
idcol = 'id'

# test_results = pd.read_csv('test_results.csv')


def modelfit(alg, dtrain, dtest, predictors, useTrainCV=True, cv_folds=30, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()

        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)

        print xgtrain.feature_names

        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='mlogloss', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['label'], eval_metric='mlogloss')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])


    # Print model report:
    print "\nModel Report"
    print "Accuracy Train : %.4g" % metrics.accuracy_score(dtrain['label'].values, dtrain_predictions)

    # #     Predict on testing data:
    dtest_predictions = alg.predict(dtest[predictors])
    print "Accuracy Test : %.4g" % metrics.accuracy_score(dtest['label'].values, dtest_predictions)

    # #     Predict on testing data:
    exact_dtest_predictions = alg.predict(exact_test_X[predictors])


    for num in exact_dtest_predictions:
        if num ==0 :
            print "functional";
        elif num ==1:
            print "non functional";
        elif num ==2:
            print "functional needs repair";
        else:
            print "******";


predictors = [x for x in train_X.columns if x not in [target, idcol]]

print predictors

xgb1 = XGBClassifier(
    learning_rate=0.01,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=6,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',
    nthread=4,
    reg_alpha=0.005,
    scale_pos_weight=1,
    num_class=3,
    seed=27)
modelfit(xgb1, train_X, test_X, predictors)
