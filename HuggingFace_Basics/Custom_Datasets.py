# Loading a custom dataset
from datasets import load_dataset

local_csv_file = load_dataset('csv',
                              data_files = 'Path to local file',  # We can also give the url to certain dataset on GitHub to open it.
                              sep = ';')

# Slicing a dicing a dataset
# Methods to split the data into training splits and shuffling them.
# Method 1::
squad = load_dataset('squad', split = 'train')
squad_shuffled = squad.shuffle(seed = 21)

#Method 2::
dataset = squad.train_test_split(test_size = 0.1)

# Getting certain words situated at specified indices
indices = [0, 10, 20, 40, 80]    # List of the indexes we need words from.
selected_words = squad.select(indices)

# Map function :: It simply applies the given function to all the columns and rows.

def lower(sent):
    return{'title':sent["title"].lower()}

squad_lower_case = squad.map(lower)

# Converting dataset into pandas dataframe
dataframe = dataset.set_format('pandas')
print(dataframe.head())
# We can also do this by an alternate method
dataframe = dataset.to_pandas()
dataframe.head()

# Saving and reloading a dataset
# Saving the dataset
raw_dataset = load_dataset('allocine')
x = raw_dataset.cache_files
# We can save dataset using any of the following
# dataset.save_to_disk(arguments)    # Used to store large datasets. Also called as Arrow data storage.
# dataset.to_csv(arguments)
# dataset.to_json(arguments)
# dataset.to_parquet(arguments)   # Used to store large datasets for longer time periods.

for split, dataset in raw_dataset.items():
    dataset.to_csv(path = f"my-dataset-{split}.csv", index = None)
# Reloading the dataset : we can simply use the load_dataset function

# Training a model from scratch takes a lot of storage and may require dataset upto some TBs or 1000s of GBs.
# To overcome this issue we use either arrow based data which directly loads the data into our hard disk or if the large is so large that it can't be fit inside our hard disk than we can use the streaming method provided by the dataset library.
# Streaming method :: It allows us to progressively load dataset by loading one element each time which creates a new data type known as IterableDataset.
# Arrow method is effective because it uses a method known as memory mapping.
# Memory mapping :: It is a mechanism that maps a portion of a file or an entire file in a disk to a chunk of virtual memory, this allows the application to access segments of extremely large file without having to read the whole file into memory first.
# This memory mapping feature of arrow makes it extremely fast when iterating over dataset.
# Streaming a large dataset : We just need to set the argument stream = True, when using the load_dataset function.
# The dataset we get from teh streaming method can't be indexed, so we can't use any indexing methods. Instead we use a method called as iter and next methods to iterate over the dataset.

large_dataset = load_dataset(path = 'csv', data_files = 'file path', split = 'train', streaming = True)
next(iter(large_dataset))

# Using map method for streamed dataset using next and iter
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-case-based')
tokenised_dataset = large_dataset.map(lambda x : tokenizer(x['text']))
next(iter(tokenised_dataset))

# We can't use the select function with streamed datasets, so we use another function called as take
dataset_head = large_dataset.take(5)    # This will give the first 5 rows from the large dataset.
dataset_skipped = large_dataset.skip(1000)    # This will skip the first 1000 rows and return the rest.


