import sys
import time
import matplotlib
from sklearn import metrics
import operator
import pandas
matplotlib.use('Agg')
import numpy as np
import xgboost as xgb
import sklearn.datasets
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from matplotlib import pylab as plt

print "Number of arguments : "+ str(sys.argv[1:].__len__())
for index in range(sys.argv[1:].__len__()):
    print str(index)+" argument value -> "+sys.argv[1:][index]
    
currTime =  long(time.time())
sbuffer = ""
if sys.argv[1:].__len__() == 5 :
    target = open(sys.argv[1:][2]+"output-"+str(currTime), 'w')
    xgb.rabit.init()
    dtrain = xgb.DMatrix(sys.argv[1:][0])
    dtest = xgb.DMatrix(sys.argv[1:][1])
    
    #Writing all features in file
    featureNamesFile = sys.argv[1:][3]+'/xgb.fmap'
    outfile = open(featureNamesFile, 'w')
    i = 0
    for featur in dtrain.feature_names:
        outfile.write('{0}\t{1}\tq\n'.format(i, featur))
        i = i + 1
    
    outfile.close()
    
    trainData, trainY = sklearn.datasets.load_svmlight_file(sys.argv[1:][0], multilabel=True, zero_based=False)
    testData, testY = sklearn.datasets.load_svmlight_file(sys.argv[1:][1], multilabel=True, zero_based=False)
    trainModY = [i[0] for i in trainY]
    testModY = [i[0] for i in testY]
    
    oneCount = 0
    zeroCount = 0
    for value in trainModY:
        if value == 0:
            zeroCount = zeroCount + 1
        else:
            oneCount = oneCount + 1
    
    scalePosWeightMin = 0
    scalePosWeightMax = 0
    if (zeroCount / oneCount) > 0 :
        scalePosWeightMax = (zeroCount / oneCount) + 5
    else:
        if (zeroCount / oneCount) - 5 < 0:
            scalePosWeightMin = 0
        else:
            scalePosWeightMin = (zeroCount / oneCount) - 5
    
    #logistics regression
    lModel = LogisticRegression()
    lModel.fit(trainData.toarray(), trainModY)
    predicted = lModel.predict(testData.toarray())
    test_preds = [1 if elem > 0.5 else 0 for elem in predicted]
    logMat =  metrics.confusion_matrix(testModY, test_preds)
    sbuffer = "Logistics Regression ==> \n"
    sbuffer += "Confusion Matrix ==> \n"
    sbuffer += "\t \t  0\t1\n"
    sbuffer += "\t0\t"+str(logMat[0][0]) +"\t"+str(logMat[0][1])+"\n"
    sbuffer += "\t1\t"+str(logMat[1][0]) +"\t"+str(logMat[1][1])+"\n"
    sbuffer += "\nPrecision : "+str(metrics.precision_score(testModY,test_preds))
    sbuffer += "\nRecall : "+str(metrics.recall_score(testModY,test_preds))
    sbuffer += "\n==============================================\n"
    
    #bayes classification
    classBayesModel = GaussianNB()
    classBayesModel.fit(trainData.toarray(), trainModY)
    predicted = classBayesModel.predict(testData.toarray())
    test_preds = [1 if elem > 0.5 else 0 for elem in predicted]
    classMat =  metrics.confusion_matrix(testModY, test_preds)
    sbuffer += "Bayes Classification ==> \n"
    sbuffer += "Confusion Matrix ==> \n"
    sbuffer += "\t \t  0\t1\n"
    sbuffer += "\t0\t"+str(classMat[0][0]) +"\t"+str(classMat[0][1])+"\n"
    sbuffer += "\t1\t"+str(classMat[1][0]) +"\t"+str(classMat[1][1])+"\n"
    sbuffer += "\nPrecision : "+str(metrics.precision_score(testModY,test_preds))
    sbuffer += "\nRecall : "+str(metrics.recall_score(testModY,test_preds))
    sbuffer += "\n==============================================\n"
    
    #bayes format
    model = BayesianRidge(compute_score=True)
    model.fit(trainData.toarray(), trainModY)
    model.fit(trainData.toarray(), trainModY)
    predicted = model.predict(testData.toarray())
    test_preds = [1 if elem > 0.5 else 0 for elem in predicted]
    bayesMat =  metrics.confusion_matrix(testModY, test_preds)
    sbuffer += "Bayes Regression ==> \n"
    sbuffer += "Confusion Matrix ==> \n"
    sbuffer += "\t \t  0\t1\n"
    sbuffer += "\t0\t"+str(bayesMat[0][0]) +"\t"+str(bayesMat[0][1])+"\n"
    sbuffer += "\t1\t"+str(bayesMat[1][0]) +"\t"+str(bayesMat[1][1])+"\n"
    sbuffer += "\nPrecision : "+str(metrics.precision_score(testModY,test_preds))
    sbuffer += "\nRecall : "+str(metrics.recall_score(testModY,test_preds))
    sbuffer += "\n==============================================\n"
    print sbuffer
    target.write(sbuffer)
    min_child_weight = 1
    while  min_child_weight <= 1:
        eta = 0.01
        while eta <= 0.3:
            depth = 6
            while depth <= 10:
                subSample = 0.5
                while subSample <= 1:
                    scalePosWeight = scalePosWeightMin
                    while scalePosWeight <= scalePosWeightMax:
                        sbuffer = ""
                        print "scalePosWeight => "+str(scalePosWeight)+" eta => "+str(eta)+" depth =>"+str(depth)+" subSample =>"+str(subSample)+" min_child_weight =>"+str(min_child_weight)
                        modelTime = long(time.time())
                        modelFileName =     sys.argv[1:][3]+"model-"+str(modelTime)
                        rawModelFileName =  sys.argv[1:][4]+"raw-model-"+str(modelTime)
                        imageFileName =     sys.argv[1:][4]+""+str(modelTime)+"-feature_importance_xgb.png"
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
                        target.writelines(sbuffer)
                        print sbuffer
                        
                        #plot feature importance graph
                        importance = bst.get_fscore(fmap=featureNamesFile)
                        print importance
                        importance = sorted(importance.items(), key=operator.itemgetter(1))
                        
                        df = pandas.DataFrame(importance, columns=['feature', 'fscore'])
                        df['fscore'] = df['fscore'] / df['fscore'].sum()
                        
                        plt.figure()
                        df.plot()
                        df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(15, 14))
                        plt.title('XGBoost Feature Importance')
                        plt.xlabel('relative importance')
                        plt.gcf().savefig(imageFileName)
                    subSample = subSample + 0.1
                depth = depth + 1
            eta = eta + 0.05
        min_child_weight = min_child_weight + 1
    xgb.rabit.finalize()
    target.close()
else :
    print "Please provide <trainfile> <testfile> <outputfile> <modelfile> <rawmodelpat>"
