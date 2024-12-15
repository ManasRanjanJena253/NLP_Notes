# Importing dependencies
from xml.sax.handler import all_features

from sklearn.feature_extraction.text import TfidfVectorizer

# Document frequency (DF) :: The number of times a word has appeared in all docs.When the DF of a word is high it means it's a generic termand need to be removed.
# Inverse Document Frequency (IDF) :: log(Total Documents/Number of documents that word is present in). This will give us a score and the word with the highest score will be valued more during or has more importance.
# Term frequency (TF) :: Total number of time term t is present in doc A/Total no. of tokens in doc A.
# TF-IDF :: TF * IDF
# We use log in IDF to decrease its effect on our TF-IDF.
# Limitation of TF-IDF ::
# 1. As our vocabulary increases, computational cost increases.
# 2. Doesn't capture the relationship between words.
# 3. Doesn't address out of vocabulary problem.

corpus = ["The aurora borealis danced across the midnight sky, mesmerizing the hikers.",
          "Professor Thompson’s eccentric mustache wiggled with excitement as he revealed the discovery",
          "The vintage typewriter on the dusty shelf whispered secrets to the curious novelist.",
          "A flock of starlings swooped down, their iridescent feathers glimmering like stained glass.",
          "The old oak tree’s gnarled branches seemed to hold the whispers of generations past."]

v = TfidfVectorizer()
transformed_output = v.fit_transform(corpus)
print(f"The vocabulary of our corpus : {v.vocabulary_}")

all_feature_names = v.get_feature_names_out()        # This function will get all the words present in the vocabulary.

for word in all_feature_names:
    index = v.vocabulary_.get(word)       # Getting the index of the word in the vocabulary.
    print(f"{word} Score : {v.idf_[index]}")   # v.idf_ will give the score of the word.

print('Transformed data :: ',transformed_output.toarray())


