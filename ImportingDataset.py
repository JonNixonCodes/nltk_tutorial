# Import a new dataset
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

from nltk.tokenize import word_tokenize

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
    
short_pos = open("short_reviews/positive.txt", encoding="latin-1").read()
short_neg = open("short_reviews/negative.txt", encoding="latin-1").read()
    
documents = []

for r in short_pos.split('\n'):
    documents.append( (r, 'pos') )

for r in short_neg.split('\n'):
    documents.append( (r, 'neg') )

# all lowercase words from movie reviews
all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())
    
# frequency distribution of all words
all_words = nltk.FreqDist(all_words)
# 5000 most common words, 5000 feature set
word_features = list(all_words.keys())[:2500]

def find_features(document):
    words = set(document) #converting a list to a set means that repeating elements are removed (term presence)
    features = {} #empty dictionary
    # features are a dictionary of the top 3000 words and whether they exist or not in the document
    for w in word_features:
        features[w] = (w in words)
        
    return features

#featuresets = [(find_features(rev), category) for(rev, category) in documents]
featuresets = []
for (rev, category) in documents:
    featuresets.append((find_features(rev), category))

#print(featuresets[:100])

random.shuffle(featuresets)
    
training_set = featuresets[:10000] #first 10000 documents
testing_set = featuresets[10000:] #remainder

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
