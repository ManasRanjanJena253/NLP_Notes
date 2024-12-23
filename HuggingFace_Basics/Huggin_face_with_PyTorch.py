# Writing training loop using PyTorch
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_dataset = load_dataset('glue', 'mrpc')
checkpoint = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation = True)
tokenized_datasets = raw_dataset.map(tokenize_function, batched = True)
data_collator = DataCollatorWithPadding(tokenizer)

# Loading the dataset into a dataloader
from torch.utils.data import DataLoader

train_dataloader = DataLoader(dataset = tokenized_datasets['train'],
                              batch_size = 8,
                              shuffle = True,
                              collate_fn = data_collator)
eval_dataloader = DataLoader(dataset = tokenized_datasets['validation'],
                             batch_size = 8,
                             collate_fn = data_collator)

# Loading a pretrained model to train on our dataset
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 2)

for batch in train_dataloader:
    break
outputs = model(**batch)

# Choosing the optimizer for our train loop
from transformers import AdamW
optimizer = AdamW(model.parameters(), lr = 5e-5)     # When training hugging face models we keep learning rates very small.

loss = outputs.loss
loss.backward()
optimizer.step()
optimizer.zero_grad()

# Scheduler : A scheduler will optimise the learning rate of our model at each step.
# Using a scheduler
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler("linear",
                             optimizer = optimizer,
                             num_warmup_steps = 0,
                             num_training_steps = num_training_steps)

# Loading our model into a gpu for faster training if gpu available.
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
print(device)

# Writing the training loop and also using the accelerator function to fasten up the process
from accelerate import Accelerator
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))
accelerator = Accelerator()    # Initiating the Accelerator
model, optimizer, train_dataloader = accelerator.prepare(model,
                                                         optimizer,
                                                         train_dataloader)
# All the usual lines of code are commented out and code with accelerator is written in the loop.

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        #loss.backward()
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)     # New way of using tqdm

# We can similarly use the accelerator in our evaluation loop
from datasets import load_metric
metric = load_metric('glue', 'mrpc')
model.eval()

eval_dataloader = accelerator.prepare(eval_dataloader)
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.inference_mode():
        outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim = -1)
        metric.add_batch(
            predictions = accelerator.gather(predictions),
            references = accelerator.gather(batch['labels'])
        )

metric.compute()




