import xgboost as xgb
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.base import TransformerMixin
from sklearn.preprocessing.label import LabelEncoder

train_df = pd.read_csv('/home/raghunandangupta/Downloads/splits/sub-splitaa.csv')
# print train_df.head(2)
# print train_df.info()

test_df = pd.read_csv('/home/raghunandangupta/Downloads/splits/sub-testtaa')
# nonnumeric_columns = ['site_id','site_domain','site_category','app_id','app_domain','app_category','device_id','device_ip','device_model']
# feature_columns_to_use = ['id','click','hour','C1','banner_pos','device_type','device_conn_type','C14','C15','C16','C17','C18','C19','C20','C21','site_id','site_domain','site_category','app_id','app_domain','app_category','device_id','device_ip','device_model']


nonnumeric_columns = ['site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
                      'device_model']
feature_columns_to_use = ['id', 'click', 'hour', 'C1', 'banner_pos', 'device_type', 'device_conn_type', 'C14', 'C15',
                          'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'site_id', 'site_domain', 'site_category', 'app_id',
                          'app_domain', 'app_category', 'device_id', 'device_model']

for fp in feature_columns_to_use:
    train_df = train_df[pd.notnull(train_df[fp])]
    test_df = test_df[pd.notnull(test_df[fp])]


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

big_test = train_df[feature_columns_to_use]
big_test_imputed = DataFrameImputer().fit_transform(big_test)

le = LabelEncoder()
for feature in nonnumeric_columns:
    big_train_imputed[feature] = le.fit_transform(big_train_imputed[feature])
    big_test_imputed[feature] = le.fit_transform(big_test_imputed[feature])

train_X = big_train_imputed[0:train_df.shape[0]]
test_X = big_test_imputed[0:test_df.shape[0]]
train_y = train_df['click']
test_Y = train_df['click']
target = 'click'
idcol = 'id'


# test_results = pd.read_csv('test_results.csv')


def modelfit(alg, dtrain, dtest, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['click'], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)

    # #     Predict on testing data:
    # dtest['predprob'] = alg.predict_proba(dtest[predictors])[:, 1]
    # results = test_results.merge(dtest[['ID', 'predprob']], on='ID')
    # print 'AUC Score (Test): %f' % metrics.roc_auc_score(results['Disbursed'], results['predprob'])


predictors = [x for x in train_X.columns if x not in [target, idcol]]
xgb1 = XGBClassifier(
    learning_rate=0.01,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=6,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    reg_alpha=0.005,
    scale_pos_weight=1,
    seed=27)
modelfit(xgb1, train_X, test_X, predictors)
