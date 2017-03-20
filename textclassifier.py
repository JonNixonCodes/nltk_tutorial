#sentiment analysis
import nltk
import random
from nltk.corpus import movie_reviews

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

# print most common words
all_words = nltk.FreqDist(all_words)
print(all_words.most_common(15))
print(all_words["stupid"])