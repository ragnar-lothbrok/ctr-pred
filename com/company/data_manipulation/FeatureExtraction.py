from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
categories = ['alt.atheism']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
count_vect = CountVectorizer()
Train_counts = count_vect.fit_transform(newsgroups_train.data)
print    count_vect.vocabulary_.get(u'man')
