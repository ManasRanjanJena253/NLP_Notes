# The trainer API : Transformers library provides a trainer api that allows us to that easily help us finetune our models on our own dataset.
# Trying this dataset on mrpc dataset
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_dataset = load_dataset('glue', 'mrpc')
checkpoint = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation = True)
tokenized_datasets = raw_dataset.map(tokenize_function, batched = True)
data_collator = DataCollatorWithPadding(tokenizer)

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 2)


from transformers import TrainingArguments
training_args = TrainingArguments(output_dir = "test_trainer",  # Inside this we can set multiple hyperparameters for our model training.
                                  per_device_train_batch_size = 16,
                                  per_device_eval_batch_size = 16,
                                  num_train_epochs = 5,
                                  learning_rate = 2e-5,
                                  weight_decay = 0.01)

from transformers import Trainer
trainer = Trainer(model,
                  training_args,
                  train_dataset = tokenized_datasets['train'],
                  eval_dataset = tokenized_datasets['validation'],
                  data_collator = data_collator,
                  tokenizer = tokenizer)

trainer.train()     # Initiating the training of our own model

predictions = trainer.predict(tokenized_datasets['validation'])
print(predictions.predictions.shape, predictions.label_ids.shape)

import numpy as np
from datasets import load_metric

metric = load_metric('glue', 'mrpc')
preds = np.argmax(predictions.predictions, axis = -1)
acc = metric.compute(predictions = preds, references = predictions.label_ids)
# The above function will give the accuracy and f1 score of our trained model.
print(acc)


