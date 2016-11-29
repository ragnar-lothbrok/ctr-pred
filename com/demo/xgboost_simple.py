import pandas as pd
import time
import matplotlib
from sklearn import metrics
import operator
matplotlib.use('Agg')
import numpy as np
import xgboost as xgb
from matplotlib import pylab as plt

scalePosWeight = 3
eta = 0.3
depth = 6
subSample=1
min_child_weight = 1

xgb.rabit.init()
dtrain = xgb.DMatrix("/home/raghunandangupta/Desktop/books/train_file_new")
dtest = xgb.DMatrix("/home/raghunandangupta/Desktop/books/test_file_new")
sbuffer = ""

print "scalePosWeight => "+str(scalePosWeight)+" eta => "+str(eta)+" depth =>"+str(depth)+" subSample =>"+str(subSample)+" min_child_weight =>"+str(min_child_weight)
modelTime = long(time.time())
modelFileName =  "/home/raghunandangupta/Desktop/books/model_file"
rawModelFileName = "/home/raghunandangupta/Desktop/books/raw_model_file"
param = {'max_depth': depth, 'eta': eta, 'silent': 1, 'objective': 'binary:logistic', 'subsample': subSample,
         'colsample_bytree': 0.8, 'alpha': 0.001, 'min_child_weight': min_child_weight, 'nthread': 16, 'scale_pos_weight':scalePosWeight}
num_round = 600

outfile = open('/home/raghunandangupta/Desktop/books/xgb.fmap', 'w')
i = 0
for featur in dtrain.feature_names:
    outfile.write('{0}\t{1}\tq\n'.format(i, featur))
    i = i + 1

outfile.close()

print ('running cross validation \n')
res = xgb.cv(param, dtrain, num_round, nfold=5,
             metrics={'logloss'}, seed=0, early_stopping_rounds=50,
             callbacks=[xgb.callback.print_evaluation(show_stdv=True)])

print res
print type(res)
print res.shape[0]

param = {'max_depth': depth, 'eta': eta, 'silent': 1, 'objective': 'binary:logistic', 'subsample': subSample,
         'colsample_bytree': 0.8, 'alpha': 0.001, 'min_child_weight': min_child_weight, 'nthread': 16, 'scale_pos_weight':scalePosWeight}
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
bst = xgb.train(param, dtrain, res.shape[0], watchlist)

test_label = dtest.get_label()
test_preds = bst.predict(dtest)

test_preds_round = [1 if elem > 0.5 else 0 for elem in test_preds]

print "\n\n=======================\n"
sbuffer += "======================================================"
sbuffer += "\nDepth "+str(depth)+" eta : "+str(eta)+" subsample :"+str(subSample)+" min_child_weight : "+str(min_child_weight)+" scale_pos_weight : "+str(scalePosWeight)+"\n"
mat =  metrics.confusion_matrix(test_label, test_preds_round)
sbuffer += "Confusion Matrix ==> \n"
sbuffer += "\t \t  0\t1\n"
sbuffer += "\t0\t"+str(mat[0][0]) +"\t"+str(mat[0][1])+"\n"
sbuffer += "\t1\t"+str(mat[1][0]) +"\t"+str(mat[1][1])+"\n"
sbuffer += 'Test error of ypred1 = \t' + str((np.sum((test_preds > 0.5) != test_label) / float(len(test_label))))+"\n"
sbuffer += "AUC Score (Test) = \t"+ str(metrics.roc_auc_score(test_label, test_preds))+"\n"
precision = metrics.precision_score(test_label, test_preds_round)
recall = metrics.recall_score(test_label, test_preds_round)
sbuffer += "Precision : "+ str(precision)+"\n"
sbuffer += "Recall :"+ str(recall)+"\n"
train_label = dtrain.get_label()
train_preds = bst.predict(dtrain)
train_preds_round = [1 if elem > 0.5 else 0 for elem in train_preds]
sbuffer += "Train error of ypred1 = \t"+ str((np.sum((train_preds > 0.5) != train_label) / float(len(train_label))))+"\n"
sbuffer += "AUC Score (Train) = \t" + str(metrics.roc_auc_score(train_label, train_preds))+"\n"
bst.save_model(modelFileName)
bst.dump_model(rawModelFileName)
sbuffer += "Model dumped into = \t"+rawModelFileName+"\n"
sbuffer += "Raw Model dumped into = \t"+rawModelFileName+"\n"
scalePosWeight = scalePosWeight + 1
print sbuffer


importance = bst.get_fscore(fmap='/home/raghunandangupta/Desktop/books/xgb.fmap')
print importance
importance = sorted(importance.items(), key=operator.itemgetter(1))


df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(15, 14))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('/home/raghunandangupta/Desktop/books/feature_importance_xgb.png')