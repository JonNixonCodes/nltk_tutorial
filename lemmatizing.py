from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("python"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("pythoning"))
#Note by default, lemmatizer assumes that the POS is noun
print(lemmatizer.lemmatize("better"))
print(lemmatizer.lemmatize("better", pos="a"))
print(lemmatizer.lemmatize("best", pos="a"))
print(lemmatizer.lemmatize("run", pos="v"))
print(lemmatizer.lemmatize("run", pos="n"))
