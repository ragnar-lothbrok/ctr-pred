import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
# from sklearn.cross_validation import train_test_split
plt.rc("font", size=14)
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

merged_matches = pandas.read_csv("train_dataset.csv", dtype={"Stadium": "category","Home Team Name": "category","Away Team Name": "category"})
current_matches = pandas.read_csv("current_dataset1.csv", dtype={"Stadium": "category","Home Team Name": "category","Away Team Name": "category"})
to_be_tested = pandas.read_csv("current_dataset.csv", dtype={"Stadium": "category","Home Team Name": "category","Away Team Name": "category"})


# sns.countplot(x='result',data=merged_matches, palette='hls')
# plt.show()
#
# sns.countplot(y="Home Team Name", data=merged_matches)
# plt.show()


sns.countplot(y="result", data=current_matches,palette='hls')
plt.show()

sns.countplot(y="Home Team Name", data=current_matches)
plt.show()

#Using Logistics regression
sns.countplot(x='result',data=merged_matches, palette='hls')
# plt.show()

sns.countplot(y="Home Team Name", data=merged_matches)
# plt.show()

merged_matches = merged_matches.apply(preprocessing.LabelEncoder().fit_transform)
current_matches = current_matches.apply(preprocessing.LabelEncoder().fit_transform)
to_be_tested = to_be_tested.apply(preprocessing.LabelEncoder().fit_transform)

merged_matches = merged_matches.drop(columns=['Stadium'])
to_be_tested = to_be_tested.drop(columns=['Stadium'])
current_matches = current_matches.drop(columns=['Stadium'])


# merged_matches.to_csv('new_dataset.csv', sep=',', encoding='utf-8')

sns.heatmap(merged_matches.corr())
# plt.show()

# Y=merged_matches['result']
# merged_matches=merged_matches.drop(columns=['result'])
#
# validation_Y=current_matches['result']
# current_matches=current_matches.drop(columns=['result'])
#
# to_be_tested = to_be_tested.drop(columns=['result'])




Y = merged_matches['result']
merged_matches = merged_matches.drop(columns=['result'])

validation_Y= current_matches['result']
current_matches=current_matches.drop(columns=['result'])


to_be_tested=to_be_tested.drop(columns=['result'])

merged_matches = pandas.get_dummies(merged_matches, columns =['Home Team Name','Away Team Name'])
current_matches = pandas.get_dummies(current_matches, columns =['Home Team Name','Away Team Name'])
to_be_tested = pandas.get_dummies(to_be_tested, columns =['Home Team Name','Away Team Name'])



classifier = LogisticRegression(random_state=0)
classifier.fit(merged_matches, Y)


print merged_matches.columns
print current_matches.columns
print to_be_tested.columns

to_be_tested = to_be_tested.drop(columns=['Away Team Name_25'])

# y_pred = classifier.predict(current_matches)

# confusion_matrix = confusion_matrix(validation_Y, y_pred)
# print(confusion_matrix)


# y_pred = classifier.predict(to_be_tested)
# for i in y_pred:
#     print i

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(merged_matches, Y)
# predictions = rf.predict(current_matches)


predictions = rf.predict(to_be_tested)
y_pred = classifier.predict(to_be_tested)

i = 0
for test in predictions:
    i = i + 1
    print test,
    if i ==37 :
        break

print "=============================="

i=0
for test in y_pred:
    i = i + 1
    print test,
    if i ==37 :
        break

print validation_Y