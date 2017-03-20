# training a classifier using scikit-learn library
import nltk
import random
from nltk.corpus import movie_reviews
#include scikit learn algos within nltk
from nltk.classify.scikitlearn import SklearnClassifier 
import pickle

# import bunch of algos
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

# more algos
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

#documents = [(list(movie_reviews.words(fileid)), category)
#             for category in movie_reviews.categories()
#             for fileid in movie_reviews.fileids(category)]

documents = []
# for positive and negative category
for category in movie_reviews.categories():
    # and each movie in this category
    for fileid in movie_reviews.fileids(category):
        # compile the (words and category) tuple to a "document" list
        documents.append((list(movie_reviews.words(fileid)), category)) #this is a tuple (a list that cannot be changed)

# shuffle all the documents in the list
random.shuffle(documents)

# print(documents[1])

# all lowercase words from movie reviews
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

# frequency distribution of all words
all_words = nltk.FreqDist(all_words)
# 3000 most common words
word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document) #converting a list to a set means that repeating elements are removed (term presence)
    features = {} #empty dictionary
    # features are a dictionary of the top 3000 words and whether they exist or not in the document
    for w in word_features:
        features[w] = (w in words)

    return features

#print(find_features(movie_reviews.words('neg/cv000_29416.txt')))

#featuresets = [(find_features(rev), category) for(rev, category) in documents]
featuresets = []
for (rev, category) in documents:
    featuresets.append((find_features(rev), category))

training_set = featuresets[:1900] #first 1900 documents
testing_set = featuresets[1900:] #last 1900 documents

# posterior = prior occurences * likelihood / evidence
#classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open('naivebayes.pickle', 'rb')
classifier = pickle.load(classifier_f)
classifier_f.close()

print("Original Naive Bayes Algo accuracy:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15) #prints most informative features

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print('MNB_classifier accuracy:', (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

#GaussianNB_classifier = SklearnClassifier(GaussianNB())
#GaussianNB_classifier.train(training_set)
#print('GaussianNB_classifier accuracy:', (nltk.classify.accuracy(GaussianNB_classifier, testing_set))*100)

BernoulliNB_classifier  = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print('BernoulliNB_classifier accuracy:', (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

LogisticRegression_classifier  = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print('LogisticRegression_classifier accuracy:', (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGD_classifier  = SklearnClassifier(SGDClassifier())
SGD_classifier.train(training_set)
print('SGD_classifier accuracy:', (nltk.classify.accuracy(SGD_classifier, testing_set))*100)

SVC_classifier  = SklearnClassifier(LinearSVC())
SVC_classifier.train(training_set)
print('SVC_classifier accuracy:', (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier  = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print('LinearSVC_classifier accuracy:', (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier  = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print('NuSVC_classifier accuracy:', (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
