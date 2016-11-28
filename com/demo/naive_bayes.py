from sklearn import metrics
import sklearn.datasets
from sklearn.naive_bayes import GaussianNB

trainData, trainY = sklearn.datasets.load_svmlight_file('/home/raghunandangupta/Desktop/books/train_file_new', multilabel=True, zero_based=False)

testData, testY = sklearn.datasets.load_svmlight_file('/home/raghunandangupta/Desktop/books/test_file_new', multilabel=True, zero_based=False)


model = GaussianNB()

trainModY = [i[0] for i in trainY]
testModY = [i[0] for i in testY]

model.fit(trainData.toarray(), trainModY)
print(model)

predicted = model.predict(testData.toarray())

print "Predicted : "+str(predicted)

# summarize the fit of the model
print(metrics.classification_report(testModY, predicted))
print(metrics.confusion_matrix(testModY, predicted))
