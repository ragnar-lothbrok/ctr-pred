import sklearn.datasets
from sklearn.linear_model import BayesianRidge
from sklearn.cross_validation import KFold, cross_val_score
from sklearn import metrics
from sklearn.grid_search import GridSearchCV

trainData, trainY = sklearn.datasets.load_svmlight_file('/home/raghunandangupta/Desktop/books/train_file_new', multilabel=True, zero_based=False)

testData, testY = sklearn.datasets.load_svmlight_file('/home/raghunandangupta/Desktop/books/test_file_new', multilabel=True, zero_based=False)

model = BayesianRidge(compute_score=True)

trainModY = [i[0] for i in trainY]
testModY = [i[0] for i in testY]

k_fold = KFold(trainData.shape[0], n_folds=20, shuffle=True, random_state=0)
print cross_val_score(model, trainData.toarray(), trainModY, cv=k_fold, n_jobs=10)
model.fit(trainData.toarray(), trainModY)
print(model)

predicted = model.predict(testData.toarray())
print "Predicted : " + str(predicted)

# summarize the fit of the model
test_preds = [1 if elem > 0.5 else 0 for elem in predicted]
print metrics.confusion_matrix(testModY,test_preds)

recall= metrics.recall_score(testModY, test_preds)
precision = metrics.precision_score(testModY, test_preds)

sbuf = str(recall)+"\n"
sbuf += str(precision)+"\n"
    
print sbuf

print "====="+str(model.get_params().keys())
param  = {'alpha_1':[1e-06],'alpha_2':[1e-06],'compute_score':[False],
            'copy_X':[True],'fit_intercept':[True],'lambda_1':[1e-06],'lambda_2':[1e-06],
            'n_iter':[300,400,500],'normalize':[False],'tol':[0.001],'verbose':[False]}
grid = GridSearchCV(estimator=model, param_grid=param)
grid.fit(trainData.toarray(), trainModY)
print(grid)
# summarize the results of the grid search
print(grid.best_score_)
    
