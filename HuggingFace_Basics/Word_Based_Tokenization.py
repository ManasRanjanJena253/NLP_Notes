# Word based tokenization means, splitting a raw text into words.
# In hugging face each separated word have an ID.
# Converting a sentence into different tokens
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("let's try to tokenize !")
print(tokens)

# Converting tokens into id's
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(token_ids)

# The tokens and the id's created with a sentence differs if padding = True, because of the attention layers present in the transformers which when converting the words into tokens also take into consideration the surrounding elements and the padding is regarded as an extra element by the attention mechanism.
# Due to this reason hugging face models use attention masks to mask the padding.


# Dataset library on hugging face : It is library used to load and use pre-made datasets
from datasets import load_dataset

raw_dataset = load_dataset('glue', 'mrpc')
print(raw_dataset)

# Using dynamic padding to increase computational speed on gpu and cpu, but it decreases computational speed in tpu.

from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer)
#train_dataloader = DataLoader(tokenized_datasets[train], batch_size = 16, shuffle = True, collate_fn = data_collator)

# We can also use of a pair of sentence for classification tasks such as if the given pair of sentences carry same meaning or not.
# To do so we just need to pass them as two arguments to the tokeniser.
checkpoint = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokens = tokenizer('My name is Manas.', 'People call me Manas')

# If we have multiple sentences to compare with another pair of multiple sentences we can create list of multiple sentences.
token = tokenizer(['My name is Manas.', 'I am Manas'],
                  ['Hello How are you ?', 'How are you?'])



