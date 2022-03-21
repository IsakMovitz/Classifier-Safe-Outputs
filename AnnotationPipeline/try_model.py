import spacy

nlp = spacy.load("./flashback_20span_keyword-model/model-last")

#nlp = spacy.load("./flashback_random_keyword-model/model-last")

doc = nlp("din jävla hora stick och brinn")
print(doc.cats)

doc = nlp("jävlar vad fint väder det är idag!")
print(doc.cats)

doc = nlp("alla judar är smuts")
print(doc.cats)

doc = nlp("jag tycker inte att det är något fel med att vara en hora")
print(doc.cats)
