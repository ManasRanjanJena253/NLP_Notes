# Tokenizing the words in a sentence using AutoTokenizer. It creates input_ids for the words in a sentence.

from transformers import AutoTokenizer

checkpoint  = "distilbert-base-uncased-finetuned-sst-2-english"   # This is the model from hugging face library that we will be using for tokenising our sentence.These models are also called as checkpoints.
tokeniser = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = ["I've been waiting for a HuggingFace course my whole life.",
          "I hate this so much!"]
inputs = tokeniser(raw_inputs, padding = True, truncation = True, return_tensors = 'pt')
# Above we have used padding = True because the raw inputs have sentenced of different lengths and during tokenization it creates numpy arrays, and each sentence is a element of that array and arrays can't have elements of different length.
# Truncation = True is set because when any sentence is provided more than the length that can be handled by the model, it needs to truncated.
# return_tensors = 'pt' tells the model to return the tensor for pytorch.

print(inputs)
# After tokenizer, it returns a dictionary with two keys, 'input_ids' containing the tokens id's of each word for each sentence of non-zero values for a word and value zero given to the padding created., 'attention_mask': contains a tensor having value as 0 or 1 , value 1 for words and 0 for the padding.

# Automodel function in hugging_face. Used to convert input_ids into logits.
# It doesn't have a classification head and thus not used for classification task.
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)

# AutoModel function for classification tasks
# It have a classification head
from transformers import AutoModelForSequenceClassification  # Used for sentiment classification.

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.logits)

# Converting the logits into probabilities
import torch.nn.functional as F

predictions = F.softmax(outputs.logits, dim = 1)
print(predictions)

# Converting predictions into corresponding labels
label_dict = model.config.id2label
print(model.config.id2label)   # This will give the labels of the classification model that we are using.
# In this the labels are in the form of a dictionary {0:'Negative', 1:'Positive'}
max_prob_1 = predictions[0].argmax()
max_prob_2 = predictions[1].argmax()

print(max_prob_1.item(), max_prob_2.item())

first_sentence_label = label_dict[max_prob_1.item()]
second_sentence_label = label_dict[max_prob_2.item()]

print(f"First Sentence Label : {first_sentence_label}")
print(f"Second Sentence Label : {second_sentence_label}")

# A pipeline function in hugging face combines all these steps and directly gives the labels of the given sentences.
# By knowing these internal function of a pipeline function we can finetune it according to our needs.



