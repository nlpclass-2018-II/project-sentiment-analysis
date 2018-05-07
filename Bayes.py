from textblob.classifiers import NaiveBayesClassifier
import string
import nltk
import sys
# import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

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
			# train_set.append(sentence.split("\t")[0])
			# buat bag of words
			train_set.append([sentence.split("\t")[0], sentence.split("\t")[1]])
			train_set_label.append(sentence.split("\t")[1])
	return train_set, train_set_label

data = load_data()	
test_set = data[0][:500]
train_set = data[0][500:]

# sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=nltk.word_tokenize)
# sklearn_representation = sklearn_tfidf.fit_transform(train_set[0])

# print(sklearn_representation)

classifier = NaiveBayesClassifier(train_set)
# acc = classifier.accuracy(train_set)
# print(acc)
string_to_classify = "i love the movie"
result = classifier.classify(string_to_classify.lower())
# classifier.show_informative_features(50)
# a  = classifier.informative_features()
# print(a)

print(classifier.accuracy(test_set))

if result == '0': print(result, "Negative Sentiment")
else: print(result,"Positive Sentiment")