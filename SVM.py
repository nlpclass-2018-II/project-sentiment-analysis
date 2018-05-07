import string
import sys
# import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
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
			# print(sentence.split("\t")[0])
	return train_set, train_set_label
data = load_data()	
train = data[0][:500]
# test = data[0][:900]
test = data[0][500:]

test_label = data[1][500:]

vectorizer = TfidfVectorizer(binary=True, use_idf=True)
	
tf_idf_train = vectorizer.fit_transform(train)
tf_idf_test = vectorizer.transform(test)

classifier = svm.SVC(kernel="linear")

classifier.fit(tf_idf_train, data[1][500:])
prediction = classifier.predict(tf_idf_test)	
score = accuracy_score(test_label, prediction)
print (tf_idf_train	)
print(score)