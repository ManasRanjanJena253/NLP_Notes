# Importing dependencies
import spacy
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Bag of n-gram words : This is a method similar to bag of words method but in this method instead of creating vocabulary using single words we use pair of words to make vocabulary. A bag of n-grams having vocabulary of only two word pair is called bi-gram.
# Bag of words is a special case of bag of n-grams where the value of n is one it can also be called unigram.
# We can combine both bag of words and bag of n-grams to get more meaningful insights from the text.
# limitations of bag of n-words:
# 1. As the n increases the dimensionality and sparsity also increases, increasing the computational cost needed.
# 2. It doesn't address out of vocabulary issue.

v = CountVectorizer(ngram_range = (2,2))     # (2,2) Means that it will only create a bi-gram we can also give (1,2) which means it will now give both bag of words and bi-gram.
v.fit(["Hello friends, I am Giovanni."])
print(f"Only bi-gram :: {v.vocabulary_}")

v = CountVectorizer(ngram_range = (1,2))
v.fit(["Hello friends, I am Giovanni."])
print(f"Both unigram and bi-gram :: {v.vocabulary_}")

corpus = ["Thor ate pizza",
          "Loki is tall",
          "Loki is eating pizza"]

# Converting the given corpus text data into a vector.
nlp = spacy.load("en_core_web_sm")

def preprocess(text : str):
    doc = nlp(text)

    filtered_tokens = []

    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)

    return (" ").join(filtered_tokens)

print(preprocess("Thor ate pizza"))

corpus_processed = [preprocess(text) for text in corpus]
print("Processed corpus :: ",corpus_processed)

v.fit(corpus_processed)
print(f"Vocabulary of our corpus :: {v.vocabulary_}")

# Converting the text into numerical vector
vector = v.transform(["Thor eat pizza"]).toarray()
print("Vectorised sentence :: ", vector)