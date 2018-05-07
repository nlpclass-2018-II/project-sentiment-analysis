from textblob.classifiers import NaiveBayesClassifier
import string
import nltk
import numpy as np
import sys
# import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
#TF IDF vectorizer
#experiment on using stop words and not using stop words

# for python2 use sys
#reload(sys)
#sys.setdefaultencoding('utf8')

train_set = []
train_set_label = []

#preprocessing, and create json file for easy access
def load_data():
	with open('imdb_labelled.txt', 'r') as f:
		for sentence in f:
			for c in string.punctuation:
				sentence = sentence.replace(c, "")
			sentence = sentence.replace("\n", "").replace("  ", "")
			sentence = sentence.lower()
			#buat tf idf
			train_set.append(sentence.split("\t")[0])
			# buat bag of words
			# train_set.append([sentence.split("\t")[0], sentence.split("\t")[1]])
			train_set_label.append(sentence.split("\t")[1])
	return train_set, train_set_label

data = load_data()	
test_set = data[0][500:]
test_set_label = data[1][500:]

train_set = data[0][:500]
train_set_label = data[1][:500]


vector = CountVectorizer(analyzer = 'word', lowercase=False)
features = vector.fit_transform(train_set)


log_model = LogisticRegression()
log_model = log_model.fit(X= features, y=train_set_label)
# print(log_model)

test_set = vector.transform(test_set)
# string_to_classify.reshape(-1,1)
result = log_model.predict(test_set)

score = accuracy_score(test_set_label, result)
print(score)