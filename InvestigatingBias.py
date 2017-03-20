# see the distribution of positive and negative accuracies
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

# import libraries required for creating voting classifier
from nltk.classify import ClassifierI
from statistics import mode

# create a new class which inherits from ClassifierI class
class VoteClassifier(ClassifierI):
    # init method always runs when class is called
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features) #either 1 or 0
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
    

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

# don't shuffle documents in the list, so we know that the early documents are negative and latter positive
# random.shuffle(documents)

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

# positive data example:
training_set = featuresets[:1900] #first 1900 documents
testing_set = featuresets[1900:] #last 100 documents (positive)

#negative data example:
training_set = featuresets[100:] #last 1900 documents
testing_set = featuresets[:100] #first 100 documents (negative)

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

LinearSVC_classifier  = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print('LinearSVC_classifier accuracy:', (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier  = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print('NuSVC_classifier accuracy:', (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  SGD_classifier,
                                  LinearSVC_classifier,
                                  NuSVC_classifier)

print('Voted_classifier accuracy:', (nltk.classify.accuracy(voted_classifier, testing_set))*100)

#print('Classification:', voted_classifier.classify(testing_set[0][0]), 'Confidence %:', voted_classifier.confidence(testing_set[0][0])*100)
#print('Classification:', voted_classifier.classify(testing_set[1][0]), 'Confidence %:', voted_classifier.confidence(testing_set[1][0])*100)
#print('Classification:', voted_classifier.classify(testing_set[2][0]), 'Confidence %:', voted_classifier.confidence(testing_set[2][0])*100)
#print('Classification:', voted_classifier.classify(testing_set[3][0]), 'Confidence %:', voted_classifier.confidence(testing_set[3][0])*100)
#print('Classification:', voted_classifier.classify(testing_set[4][0]), 'Confidence %:', voted_classifier.confidence(testing_set[4][0])*100)
#print('Classification:', voted_classifier.classify(testing_set[5][0]), 'Confidence %:', voted_classifier.confidence(testing_set[5][0])*100)
