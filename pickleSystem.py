# Import a new dataset
import nltk
import random
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

# all documents from movie reviews
documents = []
# all lowercase words from movie reviews
all_words = []

# only allow adjectives
allowed_word_types = ["J"]
for r in short_pos.split('\n'):
    documents.append( (r, 'pos') )
    pve_words = word_tokenize(r)
    pos = nltk.pos_tag(pve_words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for r in short_neg.split('\n'):
    documents.append( (r, 'neg') )
    nve_words = word_tokenize(r)
    pos = nltk.pos_tag(nve_words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

# frequency distribution of all words
all_words = nltk.FreqDist(all_words)
# 5000 most common words, 5000 feature set
word_features = list(all_words.keys())[:1500]
            
# pickle documents            
save_documents = open("jar_of_pickles/documents.pickle", "wb")
pickle.dump(documents, save_documents)
save_documents.close()

# pickle word features
save_word_features = open("jar_of_pickles/word_features2k.pickle", "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

def find_features(document):
    words = word_tokenize(document)
    features = {} #empty dictionary
    # features are a dictionary of the top 2000 words and whether they exist or not in the document
    for w in word_features:
        features[w] = (w in words)
        
    return features

#featuresets = [(find_features(rev), category) for(rev, category) in documents]
featuresets = []
for (rev, category) in documents:
    featuresets.append((find_features(rev), category))

# pickle featuresets
featuresets_f = open("jar_of_pickles/featuresets.pickle", "wb")
pickle.dump(featuresets, featuresets_f)
featuresets_f.close()

random.shuffle(featuresets)

training_set = featuresets[:10000] #first 10000 documents
testing_set = featuresets[10000:] #remainder

# posterior = prior occurences * likelihood / evidence
classifier = nltk.NaiveBayesClassifier.train(training_set)

print("Original Naive Bayes Algo accuracy:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15) #prints most informative features

## pickle classifier
save_classifier = open("jar_of_pickles/originalNaiveBayes2k.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print('MNB_classifier accuracy:', (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

## pickle classifier
save_classifier = open("jar_of_pickles/MNB_classifier2k.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier  = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print('BernoulliNB_classifier accuracy:', (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

## pickle classifier
save_classifier = open("jar_of_pickles/BernoulliNB_classifier2k.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier  = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print('LogisticRegression_classifier accuracy:', (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

## pickle classifier
save_classifier = open("jar_of_pickles/LogisticRegression_classifier2k.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

SGD_classifier  = SklearnClassifier(SGDClassifier())
SGD_classifier.train(training_set)
print('SGD_classifier accuracy:', (nltk.classify.accuracy(SGD_classifier, testing_set))*100)

## pickle classifier
save_classifier = open("jar_of_pickles/SGD_classifier2k.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

LinearSVC_classifier  = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print('LinearSVC_classifier accuracy:', (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

## pickle classifier
save_classifier = open("jar_of_pickles/LinearSVC_classifier2k.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

NuSVC_classifier  = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print('NuSVC_classifier accuracy:', (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

## pickle classifier
save_classifier = open("jar_of_pickles/NuSVC_classifier2k.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  SGD_classifier,
                                  LinearSVC_classifier,
                                  NuSVC_classifier)

print('Voted_classifier accuracy:', (nltk.classify.accuracy(voted_classifier, testing_set))*100)

def sentiment(text):
    feats = find_features(text)

    return voted_classifier.classify(feats)
