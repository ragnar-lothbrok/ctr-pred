import sys
import time
import matplotlib
from sklearn import metrics
matplotlib.use('Agg')
import numpy as np
import xgboost as xgb
import sklearn.datasets
from sklearn.linear_model import BayesianRidge

print "Number of arguments : "+ str(sys.argv[1:].__len__())
for index in range(sys.argv[1:].__len__()):
    print str(index)+" argument value -> "+sys.argv[1:][index]
    
currTime =  long(time.time())
if sys.argv[1:].__len__() == 5 :
    target = open(sys.argv[1:][2]+"output-"+str(currTime), 'w')
    xgb.rabit.init()
    dtrain = xgb.DMatrix(sys.argv[1:][0])
    dtest = xgb.DMatrix(sys.argv[1:][1])
    
    #bayes format
    trainData, trainY = sklearn.datasets.load_svmlight_file(sys.argv[1:][0], multilabel=True, zero_based=False)
    testData, testY = sklearn.datasets.load_svmlight_file(sys.argv[1:][1], multilabel=True, zero_based=False)
    trainModY = [i[0] for i in trainY]
    testModY = [i[0] for i in testY]
    model = BayesianRidge(compute_score=True)
    model.fit(trainData.toarray(), trainModY)
    model.fit(trainData.toarray(), trainModY)
    predicted = model.predict(testData.toarray())
    test_preds = [1 if elem > 0.5 else 0 for elem in predicted]
    bayesMat =  metrics.confusion_matrix(testModY, test_preds)
    target.write("Bayes Theorem ==> \n")
    target.write("Confusion Matrix ==> \n")
    target.write("\t \t  0\t1\n")
    target.write("\t0\t"+str(bayesMat[0][0]) +"\t"+str(bayesMat[0][1])+"\n")
    target.write("\t1\t"+str(bayesMat[1][0]) +"\t"+str(bayesMat[1][1])+"\n")

    model = BayesianRidge(compute_score=True)
    scalePosWeight = 0
    while  scalePosWeight <= 100:
        eta = 0.1
        while eta <= 1:
            depth = 5
            while depth <= 5:
                subSample = 0.1
                while subSample <= 1:
                    min_child_weight = 1
                    while min_child_weight <= 10:
                        print "scalePosWeight => "+str(scalePosWeight)+" eta => "+str(eta)+" depth =>"+str(depth)+" subSample =>"+str(subSample)+" min_child_weight =>"+str(min_child_weight)
                        modelTime = long(time.time())
                        modelFileName =  sys.argv[1:][3]+"model-"+str(modelTime)
                        rawModelFileName =  sys.argv[1:][4]+"raw-model-"+str(modelTime)
                        param = {'max_depth': depth, 'eta': eta, 'silent': 1, 'objective': 'binary:logistic', 'subsample': subSample,
                                 'colsample_bytree': 0.8, 'alpha': 0.001, 'min_child_weight': min_child_weight, 'nthread': 16, 'scale_pos_weight':scalePosWeight}
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
                    
                        param = {'max_depth': depth, 'eta': eta, 'silent': 1, 'objective': 'binary:logistic', 'subsample': subSample,
                                 'colsample_bytree': 0.8, 'alpha': 0.001, 'min_child_weight': min_child_weight, 'nthread': 16, 'scale_pos_weight':scalePosWeight}
                        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
                        bst = xgb.train(param, dtrain, res.shape[0], watchlist)
                    
                        test_label = dtest.get_label()
                        test_preds = bst.predict(dtest)
                    
                        test_preds_round = [1 if elem > 0.5 else 0 for elem in test_preds]
                    
                        print "\n\n=======================\n"
                        target.write("======================="+str(index)+"===============================")
                        target.write("\nDepth "+str(depth)+" eta : "+str(eta)+" subsample :"+str(subSample)+" min_child_weight : "+str(min_child_weight)+" scale_pos_weight : "+str(scalePosWeight)+"\n")
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
                        min_child_weight = min_child_weight + 1
                    subSample = subSample + 0.1
                depth = depth + 1
            eta = eta + 0.1
        scalePosWeight = scalePosWeight + 1
    xgb.rabit.finalize()
    target.close()
else :
    print "Please provide <trainfile> <testfile> <outputfile> <modelfile> <rawmodelpat>"
