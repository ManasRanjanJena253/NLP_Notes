# Importing dependencies
import spacy

# Loading our spacy model
nlp = spacy.load('en_core_web_sm')
doc = nlp("Hello ! My name is Manas. I am currently learning nlp")

for token in doc:
    print(f"Token::{token} | POS::{token.pos_} | Token Explanation::{spacy.explain(token.pos_)}")
    # POS Tagging helps us find the type of the word or what role a certain word is playing in a sentence.We can use token.tag_ to get further information about the word.
    print(f"Token Tag::{token.tag_} | Token Tag Explanation::{spacy.explain(token.tag_)}")

print(f"NLP Pipelines :: {nlp.pipe_names}")

# Removing unwanted tags from the doc
filtered_tokens = []

for token in doc:
    if token.pos_ in ["SPACE","X","Punct"]:
        filtered_tokens.append(token)

count = doc.count_by(spacy.attrs.POS)   # This gives the no. of times a tag has appeared in a doc.
print(count)
print(doc.vocab[96].text)

# NER : Named Entity Recognition. In this we try to extract entities from a
doc = nlp("Tesla is going to acquire twitter for $45 billion")
for ent in doc.ents:
    print(ent.text,"|",ent.label_,"|",spacy.explain(ent.label_))

print(f"All the entities spacy supports : {nlp.pipe_labels['ner']}")   # This will print all the entities that spacy supports

doc = nlp("Michael Bloomberg founded Bloomberg L.P in 1982")  # Here, spacy makes a mistake because it classifies Bloomberg L.P as a country or city but bloomberg is the name of an organisation. Thus, the models are not perfect.
# For better performance we can also use huggingface instead of spacy.
for ent in doc.ents:
    print(ent.text,"|",ent.label_,"|",spacy.explain(ent.label_))

# Setting a custom entity
from spacy.tokens import Span

s = Span(doc, 3, 5, label = 'ORG')
doc.set_ents([s], default = 'unmodified')

for ent in doc.ents:
    print(ent.text,"|",ent.label_,"|",spacy.explain(ent.label_))

