# When using bag of words technique to label a certain document with the word which appeared most no. of times we face the problem that many common english words such as for, is, are e.t.c. can appear same or more no. of times than the label word itself. So, to tackle this issue we need to pre-process these words beforehand. These words are called as stop words.
# We don't remove stop words when doing sentiment analysis, translation and chatbots problem.

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load("en_core_web_sm")
doc = nlp("We just opened our wings, the flying part is coming soon.")

# Getting all the stop words present in our doc
for token in doc:
    if token.is_stop:   # This checks whether the token is stop word or not.
        print(token)

# In general nlp tasks we define a function inside which all preprocessing(lemmatization, stemming, stop words removal e.t.c.) is done.
def preprocess(text : str):
    """This function takes a string as an input and does all the preprocessing required for nlp tasks."""
    doc = nlp(text)
    no_stop_words = []
    for token in doc :
        if not token.is_stop and not token.is_punct:
            no_stop_words.append(token.text)
    return (" ").join(no_stop_words)

x = preprocess("We just opened our wings, the flying part is coming soon.")
print(x)









