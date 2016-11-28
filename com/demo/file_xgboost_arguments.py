import sys
import time
import matplotlib
matplotlib.use('Agg')
import numpy as np
import xgboost as xgb
from sklearn import cross_validation, metrics
from xgboost import plot_importance
from xgboost import plot_tree
import matplotlib
import pandas as pd
import matplotlib.pylab as plt
from matplotlib import pyplot

depth = [6, 7, 8, 9, 10, 6, 7, 8, 9, 10, 6, 7, 8, 9, 10, 6, 7, 8, 9, 10]
eta = [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4]
scalePosWeight = [5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8]
subSample = [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4]
min_child_weight = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]

print "Number of arguments : "+ str(sys.argv[1:].__len__())
for index in range(sys.argv[1:].__len__()):
    print str(index)+" argument value -> "+sys.argv[1:][index]
    
currTime =  long(time.time())
if sys.argv[1:].__len__() == 5 :
    target = open(sys.argv[1:][2]+"output-"+str(currTime), 'w')
    xgb.rabit.init()
    dtrain = xgb.DMatrix(sys.argv[1:][0])
    dtest = xgb.DMatrix(sys.argv[1:][1])
    for index in range(len(depth)):
        modelTime = long(time.time())
        modelFileName =  sys.argv[1:][3]+"model-"+str(modelTime)
        rawModelFileName =  sys.argv[1:][4]+"raw-model-"+str(modelTime)
        param = {'max_depth': depth[index], 'eta': eta[index], 'silent': 1, 'objective': 'binary:logistic', 'subsample': subSample[index],
                 'colsample_bytree': 0.8, 'alpha': 0.001, 'min_child_weight': min_child_weight[index], 'nthread': 16, 'scale_pos_weight':scalePosWeight[index]}
        num_round = 600
    
        print ('running cross validation \n')
        # do cross validation, this will print result out as
        # [iteration]  metric_name:mean_value+std_value
        # std_value is standard deviation of the metric
        res = xgb.cv(param, dtrain, num_round, nfold=5,
                     metrics={'logloss'}, seed=0, early_stopping_rounds=50,
                     callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
    
        print res
        print type(res)
        print res.shape[0]
    
        param = {'max_depth': depth[index], 'eta': eta[index], 'silent': 1, 'objective': 'binary:logistic', 'subsample': subSample[index],
                 'colsample_bytree': 0.8, 'alpha': 0.001, 'min_child_weight': min_child_weight[index], 'nthread': 16, 'scale_pos_weight':scalePosWeight[index]}
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        bst = xgb.train(param, dtrain, res.shape[0], watchlist)
    
        test_label = dtest.get_label()
        test_preds = bst.predict(dtest)
    
        test_preds_round = [1 if elem > 0.5 else 0 for elem in test_preds]
    
        print "\n\n=======================\n"
        target.write("======================="+str(index)+"===============================")
        target.write("\nDepth "+str(depth[index])+" eta : "+str(eta[index])+" subsample :"+str(subSample[index])+" min_child_weight : "+str(min_child_weight[index])+" scale_pos_weight : "+str(scalePosWeight[index])+"\n")
        mat =  metrics.confusion_matrix(test_label, test_preds_round)
        target.write("Confusion Matrix ==> \n")
        target.write("\t \t  0\t1\n")
        target.write("\t0\t"+str(mat[0][0]) +"\t"+str(mat[0][1])+"\n")
        target.write("\t1\t"+str(mat[1][0]) +"\t"+str(mat[1][1])+"\n")
        target.write ('Test error of ypred1 = \t' + str((np.sum((test_preds > 0.5) != test_label) / float(len(test_label))))+"\n")
        target.write( "AUC Score (Test) = \t"+ str(metrics.roc_auc_score(test_label, test_preds))+"\n")
        target.write( "Recall : "+ str(metrics.precision_score(test_label, test_preds_round))+"\n")
        target.write( "Precision :"+ str(metrics.recall_score(test_label, test_preds_round))+"\n")
        train_label = dtrain.get_label()
        train_preds = bst.predict(dtrain)
        train_preds_round = [1 if elem > 0.5 else 0 for elem in train_preds]
        target.write ("Train error of ypred1 = \t"+ str((np.sum((train_preds > 0.5) != train_label) / float(len(train_label))))+"\n")
        target.write( "AUC Score (Train) = \t" + str(metrics.roc_auc_score(train_label, train_preds))+"\n")
        bst.save_model(modelFileName)
        bst.dump_model(rawModelFileName)
        target.write("Model dumped into = \t"+rawModelFileName+"\n")
        target.write("Raw Model dumped into = \t"+rawModelFileName+"\n")
    xgb.rabit.finalize()
    target.close()
else :
    print "Please provide <trainfile> <testfile> <outputfile> <modelfile> <rawmodelpat>"
