# Why use word embeddings ?
# It gives similar vectors to similar words, making it easier to find sentences having same meanings.
# Dimensions are low.
# The various word embedding techniques ::
# 1. Word2Vec
# 2. GloVe
# 3. fastText
# Based on transformer architecture ::
# 1. BERT
# 2. GPT
# Based on LSTM ::
# 1. ELMo
# Word2Vec has a special feature to perform arithmetic on words to get a new word. For eg :: King - man + woman = Queen.

# Using word embedding methods using spacy
# Word vectors occupy lot of space. Hence en_core_web_sm model do not have them included.
# In order to download word vectors, we need to install large or medium english model. We will install the large model.

import spacy

nlp = spacy.load("en_core_web_lg")
doc = nlp("Dog ate a banana, Now he is happy.")
for token in doc :
    print(token.text, "Vector :", token.has_vector, "OOV :", token.is_oov)  # OOV means out of vocabulary.

# Checking the similarity between two words in a doc using spacy large english model.
base_tokens = ["Sandwich", "Grains", "Oats"]
doc = nlp("Bread is made of wheat.")

# Checking the similarity of words in base tokens with the words in the doc

l = []
for k in base_tokens:
    k = nlp(k)
    for token in doc:
        print(f"Similarity of '{token.text}' with base token '{k}' is :: {token.similarity(k)}")
        if token.similarity(k) > 0.6:
            l.append(f"{token.text} : {k.text} : {round(token.similarity(k), 2)}%")
print(f"Tokens with highest similarity with the base_tokens :: {l}")