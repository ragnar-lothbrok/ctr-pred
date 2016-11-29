from sklearn import metrics
import sklearn.datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.grid_search import GridSearchCV

trainData, trainY = sklearn.datasets.load_svmlight_file('/home/raghunandangupta/Desktop/books/train_file_new', multilabel=True, zero_based=False)

testData, testY = sklearn.datasets.load_svmlight_file('/home/raghunandangupta/Desktop/books/test_file_new', multilabel=True, zero_based=False)

trainModY = [i[0] for i in trainY]
testModY = [i[0] for i in testY]

model = GaussianNB()

k_fold = KFold(trainData.shape[0], n_folds=10, shuffle=True, random_state=0)
print cross_val_score(model, trainData.toarray(), trainModY, cv=k_fold, n_jobs=10)
print "print it"

model.fit(trainData.toarray(), trainModY)
print(model)

predicted = model.predict(testData.toarray())

print "Predicted : " + str(predicted)

# summarize the fit of the model
print(metrics.classification_report(testModY, predicted))
print(metrics.confusion_matrix(testModY, predicted))
