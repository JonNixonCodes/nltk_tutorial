from nltk.corpus import wordnet

syns = wordnet.synsets("program")

print(syns)

#synset
print(syns[1])

#just the word
print(syns[1].lemmas()[0].name())

#definition
print(syns[1].definition())

#examples
print(syns[1].examples())

synonyms = []
antonyms = []

#why is trade_good a lemma for commodity
for syn in wordnet.synsets("good"):
    print("syn:", syn)
    for l in syn.lemmas():
        print("l:", l)
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

#print(set(synonyms))
#print(set(antonyms))

#compare semantic similarity using wup(Wu and Palmer)
w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")
print(w1.wup_similarity(w2))

w3 = wordnet.synset("car.n.01")
print(w3.wup_similarity(w1))

w4 = wordnet.synset("cactus.n.01")
print(w4.wup_similarity(w1))

w5 = wordnet.synset("cat.n.01")
print(w5.wup_similarity(w1))

w6 = wordnet.synset("positive.n.01")
print(w6.wup_similarity(w1))