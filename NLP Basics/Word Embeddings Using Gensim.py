# Gensim is python library used for nlp tasks specifically to do topic modelling.
import gensim.downloader as api

wv = api.load("word2vec-google-news-300")  # Code to download word2vec model which is trained on Google News articles, It contains more than 100 billion words and has a size of 1.6GB.

# Checking the similarity between two words
word1 = 'great'
word2 = 'good'
print(f"The similarity between {word1} and {word2} :: {wv.similarity(w1 = word1, w2 = word2)}")

most_similar = wv.most_similar("good")  # This function will give a list of words which are most similar to the word good and how similar they are.
print(most_similar)

# Doing arithmetic using gensim
# king - man + woman
x = wv.most_similar(positive = ['king', 'woman'], negative = ['man'])    # Gives a list of words having similarity same as the word equation provided.
print(x)

# Finding dissimilarity between various words
dissimilar_word = wv.doesnt_match(['dog', 'cat', 'lion', 'microsoft'])
print(dissimilar_word)

# When finding the vector representation of a whole sentence we can use average method through which we get the average of all the word vector present in the sentence, and now this averaged vector represents the sentence.
# For eg :: If a sentence contains two words i.e. Hello dog. It's vector representation will be::
sentence_vector = wv.get_mean_vector(["Hello", "dog"])
