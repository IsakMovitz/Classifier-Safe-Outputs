import spacy

nlp = spacy.load("sexist-model/model-best")

doc = nlp("knulla kåt porr prostituerad")
print(doc.cats)

doc = nlp("massage escort eskort milf")
print(doc.cats)

doc = nlp("jag är en snäll katt")
print(doc.cats)

doc = nlp("jag gillar potatismos")
print(doc.cats)
