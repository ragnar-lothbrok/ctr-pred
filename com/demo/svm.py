from sklearn import metrics
from sklearn.svm import SVC
import sklearn.datasets

# load the iris datasets
trainData, trainY = sklearn.datasets.load_svmlight_file('/home/raghunandangupta/Desktop/books/train_file_new', multilabel=True, zero_based=False)
# fit a SVM model to the trainData
model = SVC()
trainModX = [i[0] for i in trainY]
model.fit(trainData.toarray(), trainModX)
print(model)
# make predictions
predicted = model.predict(trainData.toarray())
# summarize the fit of the model
print(metrics.classification_report(trainModX, predicted))
print(metrics.confusion_matrix(trainModX, predicted))
