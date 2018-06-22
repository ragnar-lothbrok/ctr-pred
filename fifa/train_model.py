import numpy as np
import xgboost as xgb
from sklearn import metrics
from matplotlib import pylab as plt
import operator
import pandas
import matplotlib
import io

dictionary  ={'f1':'bank_operation_code','f2':'beneficiary','f4':'month','f5':'year','f6':'ordering_customer',
              'f7':'ordering_institution'}

def create_feature_map(features):
	outfile = open('xgb.fmap', 'w')
	i = 0
	for feat in features:
		outfile.write('{0}\t{1}\tq\n'.format(i, feat))
		i = i + 1

	outfile.close()

def convert(x):
    if x > 0.7 :
        return 2
    elif x < .7 and x > 0.3:
        return 1
    else:
        return 0

dtrain = xgb.DMatrix('csvexample3.csv?format=csv&label_column=0')
dtest = xgb.DMatrix('testing.csv?format=csv&label_column=0')
param = {'max_depth': 5, 'eta': 0.15, 'silent': 1, 'objective': 'reg:logistic', 'subsample': 0.8,
         'colsample_bytree': 0.8, 'alpha': 0.001, 'min_child_weight': 1, 'nthread': 8}
num_round = 400

print ('running cross validation')
# do cross validation, this will print result out as
# [iteration]  metric_name:mean_value+std_value
# std_value is standard deviation of the metric
res = xgb.cv(param, dtrain, num_round, nfold=8,
             metrics={'rmse'}, seed=0, early_stopping_rounds=100,
             callbacks=[xgb.callback.print_evaluation(show_stdv=True)])

print(res)
print(res.shape[0])

param = {'max_depth': 8, 'eta': 0.15, 'silent': 1, 'objective': 'reg:linear', 'subsample': 0.8,
         'colsample_bytree': 0.8, 'alpha': 0.001, 'min_child_weight': 1, 'nthread': 8,
         'eval_metric': 'rmse'}
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
bst = xgb.train(param, dtrain, res.shape[0], watchlist)

test_label = dtest.get_label()
test_preds = bst.predict(dtest)

for elem in test_preds:
    print(elem)

test_preds_round = [convert(i) for i in test_preds]
test_label_new = [convert(i) for i in test_label]

print(metrics.confusion_matrix(test_label_new, test_preds_round))

xgb.plot_importance(bst)

xgb.plot_tree(bst, num_trees=2)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(150, 100)
fig.savefig('tree.png')

importance = bst.get_fscore()
if importance.items().__len__() > 0 :
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    featureTuples = []
    for typ in importance:
        featureTuples.append((dictionary.get(typ[0]),typ[1]))
    print(featureTuples)
    df = pandas.DataFrame(featureTuples, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    plt.figure()
    df.plot()
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(20, 14))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    plt.gcf().savefig('test.png')
    print('printed')

print('Test error of ypred1=%f' % (np.sum((test_preds > 0.5) != test_label) / float(len(test_label))))

train_label = dtrain.get_label()
train_preds = bst.predict(dtrain)

print('Train error of ypred1=%f' % (np.sum((train_preds > 0.5) != train_label) / float(len(train_label))))
