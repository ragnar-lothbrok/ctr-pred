import numpy as np
import xgboost as xgb
from sklearn import metrics
import scipy

weight = 4
dtrain = xgb.DMatrix('/home/raghunandangupta/Downloads/soc_gen_data/abc.txt')
print dtrain.get_label()
dtest = xgb.DMatrix('/home/raghunandangupta/Downloads/soc_gen_data/test/part-00000')
param = {'max_depth': 7, 'eta': 0.15, 'silent': 1, 'objective': 'binary:logistic', 'subsample': 0.8,
         'colsample_bytree': 0.8, 'alpha': 0.001, 'min_child_weight': 1, 'nthread': 8,'scale_pos_weight':weight}
num_round = 400

print ('running cross validation')
# do cross validation, this will print result out as
# [iteration]  metric_name:mean_value+std_value
# std_value is standard deviation of the metric
res = xgb.cv(param, dtrain, num_round, nfold=5,
             metrics={'logloss'}, seed=0, early_stopping_rounds=50,
             callbacks=[xgb.callback.print_evaluation(show_stdv=True)])

print res
print res.shape[0]

param = {'max_depth': 7, 'eta': 0.15, 'silent': 1, 'objective': 'binary:logistic', 'subsample': 0.8,
         'colsample_bytree': 0.8, 'alpha': 0.001, 'min_child_weight': 1, 'nthread': 8,
         'eval_metric': 'logloss','scale_pos_weight':weight}
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
bst = xgb.train(param, dtrain, res.shape[0], watchlist)

test_label = dtest.get_label()
test_preds = bst.predict(dtest)

print test_preds

test_preds_round = [1 if elem > 0.5 else -1 for elem in test_preds]
for cluster in test_preds_round:
    print cluster

# test_preds_round = [1 if elem > 0.5 else 0 for elem in test_preds]
# 
# print metrics.confusion_matrix(test_label, test_preds_round)
# print ('Test error of ypred1=%f' % (np.sum((test_preds > 0.5) != test_label) / float(len(test_label))))
# 
# train_label = dtrain.get_label()
# train_preds = bst.predict(dtrain)
# 
# print ('Train error of ypred1=%f' % (np.sum((train_preds > 0.5) != train_label) / float(len(train_label))))