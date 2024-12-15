# Representing text as a vector is also known as Vector Space Model.
# Various approaches of converting text into vector ::
# 1. One Hot Encoding
# 2. Bag of Words
# 3. TF-IDF
# 4. Word Embeddings
# There are also many more methods but these are the most commonly used.
# In nlp, feeding a good text representation to an ordinary algorithm will get you much better result than compared to applying a top-notch algorithm to an ordinary text representation.
from random import random

# Label Encoding : In label encoding we give numbers to all the words that a certain document have and whenever that word appears it is replaced with that number to create a vector.
# Hot label encoding is not as effective because it can't compare the similarity of the words. It also consumes a lot of memory and need a lot of computation power. It also suffers from out of vocabulary problem.
# Both label encoding and hot label encoding are not suitable methods of making vectors from texts.

# Bag of Words : In this method we count the no. of times a certain word has appeared and based on this we will be categorising our data on the basis of which word has appeared more frequently.
# Vocabulary : It is a list of all the unique words appeared in a document.
# Limitation of bag of words : Our vocabulary may be long and thus may require more computation power, but it's still better than hot label encoding.

# Using Bag of Words
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv("spam.csv")
print(df.head())
df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)                 # Category is the column telling if mail is spam or not.
# The above line of code will create a new column named as spam and if that row has spam email the value of that column will be 1 otherwise 0.

x_train, x_test, y_train, y_test = train_test_split(x = df['message'], y = df['spam'], stratify = df['spam'], random_state = 21, test_size = 0.2 )

from sklearn.feature_extraction.text import CountVectorizer      # This will create a vocabulary to be used by our Bag of Words.

v = CountVectorizer()
x_train_cv = v.fit_transform(x_train.values)    # This will result in a datatype of sparse matrix type, which we need to convert into numpy array.
x_train_np = x_train_cv.toarray()

print(v.get_feature_names_out())   # This function give the whole vocabulary.

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train_cv, y_train)
y_pred = model.predict(v.fit_transform(x_test))

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))   # Classification report will give use all the model performance metrics calculated, eg : precision, recall, f1-score, support.

# Using the pipeline method from sklearn to do all these steps with a single line of code.
from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

clf.fit(x_train, y_train)










