import spacy

nlp = spacy.blank('en')    # Creating a blank english language nlp pipeline.
doc = nlp("Hello friends my name is Manas Ranjan Jena. I study in Punjab Engineering College.")

for token in doc:   # This has already done word tokenization as soon as we wrote the sentences into the doc variable.
    print(token)
# We can use indexing to access each word.
print(f"First word :: {doc[0]}")

# Type of these objects
print(f"Type of doc :: {type(doc)}")
print(f"Type of the tokens :: {type(token)}")

# Span object : It is the type assigned when an object is extracted from the doc by using indexing.
span = doc[0:5]   # Span type object.
print(f"Type of span :: {type(span)}")

# Using some methods made for tokens
token = doc[0]
print(token.is_alpha)   # This give a boolean as an output. If the given token is alphabet or not.
print(token.like_num)   # This checks whether the given token is alphabet or number.

# When the like_num method is used with a string but the string is the spelling of a number it would return true.

print(token.is_currency)   # Tells whether the provided token is a currency or not.
print(token.like_email)   # Tells whether the provided token is an email or not.

# Customising spacy according to our special needs
from spacy.symbols import ORTH

nlp.tokenizer.add_special_case("Gimme", [{ORTH : "Gim"},
                                         {ORTH : "me"}])  # Whenever the slang gimme is used spacy will now return two tokens give and me.

slang = nlp("Gimme that pen")

for token in slang:
    print(token)

print(f"Pipeline names before loading a pipeline : {nlp.pipe_names}")   # This gives the names of the pipelines that we are currently using. Currently this will result in an empty list as we are using a blank english pipeline.

# Loading a nlp pipeline
nlp = spacy.load("en_core_web_sm")
print(f"Pipeline names after loading a pipeline : {nlp.pipe_names}")

doc = nlp("Hello friend, How are you ?")

# POS stands for part of speech. It tells what is the function of that word in that sentence(Is the given token an adjective, noun, punctuation e.t.c)
# Lemmatisation means converting the word into its root word. For eg : Root word of running is run.
for token in doc:
    print(f"Token:{token} | POS:{token.pos_} | Lematised word:{token.lemma_}")
    print()

# Recognising the entities using spacy
doc = nlp("Tesla is going to acquire twitter which is worth $45 billion")

for ent in doc.ents:
    print(f"Entity::{ent} | Entity Label::{ent.label_}")

# Changing the visual of the above entity displaying method

from spacy import displacy

print(displacy.render(doc, style = 'ent'))

# Stemming : Using fixed rules such as remove able, ing etc. to derive base word is called stemming.
# Lemmatization : Use knowledge of a language to derive a base word.

# Stemming
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
words = ["eating", "eats", "ate", "adjustable", "ability", "meeting"]

print("-----------------------------------Stemming------------------------------------")

for word in words:
    print(word, "|", stemmer.stem(word))

# Lemmatization
nlp = spacy.load("en_core_web_sm")

print("-----------------------------------Lemmatization------------------------------")

for token in doc:
    print(token, "|", token.lemma_)

# Tuning the rules so that our model can classify the slang brudda as brother.

ar = nlp.get_pipe('attribute_ruler')
ar.add([[{"TEXT" : "Brudda"}]], {"LEMMA" : "Brother"})

doc = nlp("Brudda how are you ??")
for token in doc:
    print(token, "|", token.lemma_)
