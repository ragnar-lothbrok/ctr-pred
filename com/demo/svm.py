from sklearn import metrics
from sklearn.svm import SVC
import sklearn.datasets

# load the iris datasets
data, outcome = sklearn.datasets.load_svmlight_file('/home/raghunandangupta/Desktop/books/train_file_new', multilabel=True, zero_based=False)
# fit a SVM model to the data
model = SVC()
expected = [i[0] for i in outcome]
model.fit(data.toarray(), expected)
print(model)
# make predictions
predicted = model.predict(data.toarray())
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
