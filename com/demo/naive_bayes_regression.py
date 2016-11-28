import sklearn.datasets
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import metrics

trainData, trainY = sklearn.datasets.load_svmlight_file('/home/raghunandangupta/Desktop/books/train_file_new', multilabel=True, zero_based=False)

testData, testY = sklearn.datasets.load_svmlight_file('/home/raghunandangupta/Desktop/books/test_file_new', multilabel=True, zero_based=False)

model = BayesianRidge(compute_score=True)

trainModY = [i[0] for i in trainY]
testModY = [i[0] for i in testY]

model.fit(trainData.toarray(), trainModY)
print(model)

predicted = model.predict(testData.toarray())
print "Predicted : " + str(predicted)

# summarize the fit of the model
test_preds = [1 if elem > 0.5 else 0 for elem in predicted]
print metrics.confusion_matrix(testModY,test_preds)

