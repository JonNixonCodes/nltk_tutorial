import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

#print location of nltk init file
print(nltk.__file__)

#some texts are loaded as raw text
sample = gutenberg.raw("bible-kjv.txt")

sentences = sent_tokenize(sample)

print(sentences[5:15])