#https://xgboost.readthedocs.io/en/latest/build.html
import numpy as np
import xgboost as xgb
from sklearn import cross_validation, metrics
from xgboost import plot_importance
from matplotlib import pyplot

xgb.rabit.init()
dtrain = xgb.DMatrix('/tmp/clicks')
dtest = xgb.DMatrix('/tmp/clicks_test')
param = {'max_depth': 7, 'eta': 0.2, 'silent': 1, 'objective': 'binary:logistic', 'subsample': 0.8,
         'colsample_bytree': 0.8, 'alpha': 0.001, 'min_child_weight': 1, 'nthread': 8, 'scale_pos_weight':50}
num_round = 100

print ('running cross validation')
# do cross validation, this will print result out as
# [iteration]  metric_name:mean_value+std_value
# std_value is standard deviation of the metric
res = xgb.cv(param, dtrain, num_round, nfold=5,
             metrics={'logloss'}, seed=0, early_stopping_rounds=50,
             callbacks=[xgb.callback.print_evaluation(show_stdv=True)])

print res
print type(res)
print res.shape[0]

param = {'max_depth':7, 'eta': 0.2, 'silent': 1, 'objective': 'binary:logistic', 'subsample': 0.8,
         'colsample_bytree': 0.8, 'alpha': 0.001, 'min_child_weight': 1, 'nthread': 8,
         'eval_metric': ['auc','logloss'], 'scale_pos_weight':50}
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
bst = xgb.train(param, dtrain, res.shape[0], watchlist)

test_label = dtest.get_label()
test_preds = bst.predict(dtest)

test_preds_round = [1 if elem > 0.5 else 0 for elem in test_preds]

print metrics.confusion_matrix(test_label, test_preds_round)
print ('Test error of ypred1=%f' % (np.sum((test_preds > 0.5) != test_label) / float(len(test_label))))
print "AUC Score (Test): %f" % metrics.roc_auc_score(test_label, test_preds)
print metrics.precision_score(test_label, test_preds_round)

train_label = dtrain.get_label()
train_preds = bst.predict(dtrain)
train_preds_round = [1 if elem > 0.5 else 0 for elem in train_preds]

print ('Train error of ypred1=%f' % (np.sum((train_preds > 0.5) != train_label) / float(len(train_label))))
print "AUC Score (Train): %f" % metrics.roc_auc_score(train_label, train_preds)
bst.save_model('/tmp/001.model')
bst.dump_model('/tmp/001.raw.txt')
xgb.rabit.finalize()
plot_importance(bst)
pyplot.show()