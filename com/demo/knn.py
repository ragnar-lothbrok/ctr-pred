from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import sklearn.datasets

trainData, trainY = sklearn.datasets.load_svmlight_file('/home/raghunandangupta/Desktop/books/train_file_new', multilabel=True, zero_based=False)
model = KNeighborsClassifier()
trainModX = [i[0] for i in trainY]
model.fit(trainData.toarray(), trainModX)
# make predictions
predicted = model.predict(trainData.toarray())
# summarize the fit of the model
print(metrics.classification_report(trainModX, predicted))
print(metrics.confusion_matrix(trainModX, predicted))
