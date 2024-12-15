# In fastText we use a technique called as Character n gram where n is a hyperparameter based on which we will be taking the parts of a word.
# For eg :: If n = 3, and the word is 'capable' The word will be divided into cap, apa, pab, abl, ble.
# The benefit of doing so is that it captures for fine or granular details about the word. It now can also address the issue of out of vocabulary as most words are derived from their root words and have approximately same meaning and matching just a pair of words instead of the whole word help us also turning the out of vocabulary words into vectors.
# fastText is often a first choice when you want to train custom word embeddings for you own domain.
# Downloading the fasttext libraries requires a lot of space, So all the codes in this section are not running.

import fasttext
model_en = fasttext.load_model("File path where the model is downloaded and saved.")
model_en.get_nearest_neighbors("good")   # This will give a list of most similar words to the provided with their similarity percentage.
model_en.get_word_vector('Good')   # Getting the vector form of the given word.
model_en.get_analogies("berlin", "germany", "India")   # This code will try t establish a relation between the given set of words and give a list of words which might be similar to the relation established.

# Training our own fasttext model
model = fasttext.train_unsupervised("This will require a csv file containing only text which is already preprocessed(Removed extra spaces, every character in same case, removing newline characters.")
# After training, we can use all the previously used method on our trained model and then use it.
