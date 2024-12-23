# The pretrained tokenizer might produce anomalies and may not be suitable if we are using it for following reasons ::
# 1. New language, that our tokenizer is not trained on.
# 2. New characters.
# 3. New Style.
# 4. New domain. i.e. A model trained on english vocabulary won't perform well on medical documents and medical terms.

# Steps for training a new tokenizer :
# loading the training corpus(dataset)
from datasets import load_dataset
raw_dataset = load_dataset('code_search_net', 'python', trust_remote_code = True)

def get_training_corpus():
    dataset = raw_dataset['train']
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples['whole_func_string']

from transformers import AutoTokenizer
training_corpus = get_training_corpus()
old_tokenizer = AutoTokenizer.from_pretrained('gpt2')
new_tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)

# Saving the new tokenizer
new_tokenizer.save_pretrained('Newly_trained_model')