from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import sklearn.datasets

trainData, trainY = sklearn.datasets.load_svmlight_file('/home/raghunandangupta/Desktop/books/train_file_new', multilabel=True, zero_based=False)
model = LogisticRegression()
trainModX = [i[0] for i in trainY]
model.fit(trainData.toarray(), trainModX)
# print(model)

predicted = model.predict(trainData.toarray())

print predicted
print trainModX

# summarize the fit of the model
print(metrics.classification_report(trainModX, predicted))
print(metrics.confusion_matrix(trainModX, predicted))
print metrics.precision_score(trainModX, predicted)
print metrics.recall_score(trainModX, predicted)