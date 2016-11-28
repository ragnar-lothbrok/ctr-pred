import numpy as np
import xgboost as xgb
from sklearn import cross_validation, metrics
from xgboost import plot_importance
from xgboost import plot_tree
import matplotlib
import pandas as pd
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

from matplotlib import pyplot
xgb.rabit.init()
dtrain = xgb.DMatrix('/tmp/click_impression_20161115/lol/train_file_new')
dtest = xgb.DMatrix('/tmp/click_impression_20161115/lol/test_file_new')

depth = [6, 7, 8, 9, 10, 6, 7, 8, 9, 10, 6, 7, 8, 9, 10, 6, 7, 8, 9, 10]
eta = [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4]
scalePosWeight = [5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 15, 15, 15, 15, 15, 8, 8, 8, 8, 8]
subSample = [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4]
min_child_weight = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]
for index in range(len(depth)):
    print index
    param = {'max_depth': depth[index], 'eta': eta[index], 'silent': 0, 'objective': 'binary:logistic', 'subsample': subSample[index],
             'colsample_bytree': 0.8, 'alpha': 0.001, 'min_child_weight': min_child_weight[index], 'nthread': 16, 'scale_pos_weight':scalePosWeight[index]}
    num_round = 600
    
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
    
    param = {'max_depth': depth[index], 'eta': eta[index], 'silent': 0, 'objective': 'binary:logistic', 'subsample': subSample[index],
             'colsample_bytree': 0.8, 'alpha': 0.001, 'min_child_weight': min_child_weight[index], 'nthread': 16, 'scale_pos_weight':scalePosWeight[index]}
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    bst = xgb.train(param, dtrain, res.shape[0], watchlist)
    
    test_label = dtest.get_label()
    test_preds = bst.predict(dtest)
    
    test_preds_round = [1 if elem > 0.5 else 0 for elem in test_preds]
    
    print metrics.confusion_matrix(test_label, test_preds_round)
    print ('Test error of ypred1=%f' % (np.sum((test_preds > 0.5) != test_label) / float(len(test_label))))
    print "AUC Score (Test): %f" % metrics.roc_auc_score(test_label, test_preds)
    print metrics.precision_score(test_label, test_preds_round)
    print metrics.recall_score(test_label, test_preds_round)
    
    train_label = dtrain.get_label()
    train_preds = bst.predict(dtrain)
    train_preds_round = [1 if elem > 0.5 else 0 for elem in train_preds]
    
    print ('Train error of ypred1=%f' % (np.sum((train_preds > 0.5) != train_label) / float(len(train_label))))
    print "AUC Score (Train): %f" % metrics.roc_auc_score(train_label, train_preds)
    bst.save_model('/apps/learning/xgb.model')
    bst.dump_model('/apps/learning/xgb.raw.txt')
    feat_imp = pd.Series(bst.get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    xgb.rabit.finalize()