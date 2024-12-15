# In Regex, we use pattern matching to extract important data from a conversation.
import re
# We can use regular expression to find certain word combination using patterns.
# re.findall(pattern, text)   this is the function to find all the occurrence of a particular pattern in the passed text.

text = "Hello my name is Manas Ranjan Jena."
# Let's assume we need to extract the name from this text.We can see the name always occurs after the word name.
pattern = 'name is(.*).'

extracted_name = re.findall(pattern, text)
print(f"The names extracted from the text are :: {extracted_name}")  # Returns a list with all the words matching the given pattern.

# Spacy and Nltk
# The main difference between nltk and spacy is that spacy is object orientes whereas nltk is string processing.
# Nltk need to be fine-tuned to be beneficial for doing a specific task whereas spacy is more user-friendly and don't need to have any changes done to it and works well without any fine-tuning.

import spacy

nlp = spacy.load("en_core_web_sm")  # Loading the type of spacy we want.Here we want spacy to perform operations on english language which is given by en part of the string that we passed.
doc = nlp("Dr.Strange loves pav bhaji of mumbai. Hulk loves chaat of delhi.")  # Creating a document on which our we will be performing operations using spacy.

# Splitting the different sentences in a given document or string using spacy. This process of separating out different sentences from a given document is called sentence tokenization.
print(f"Sentences :: {doc.sents}")  # This will result in a sort of dtype. We need to use loop to access various sentences divided by spacy.
for sentence in doc.sents:
    print(sentence)

# Splitting the different sentences in a given document or string using spacy. This process of separating out different words from a given document is called word tokenization.
for sentence in doc.sents:
    for word in sentence:
        print(word)

# Doing the same above tasks using nltk
import nltk
from nltk.tokenize import sent_tokenize     # Chossing the type of nltk we want to use.

x = sent_tokenize("Dr.Strange loves pav bhaji of mumbai. Hulk loves chaat of delhi.")
print(x)