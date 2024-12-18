# Configuration of a model is a blueprint that contain all the essentials for building the model architecture.
# A few methods to configure a pretrained model
# 1st Method : Using autoconfig
from transformers import AutoConfig

bert_config1 = AutoConfig.from_pretrained('bert-base-cased')
gpt_config = AutoConfig.from_pretrained('gpt2')
bart_config = AutoConfig.from_pretrained('facebook/bart-base')
print(f"The bert_config : {bert_config1}")
print('-------------------------------------------------------------------------------')
print(f"The gpt_config : {gpt_config}")
print('-------------------------------------------------------------------------------')
print(f"The bart_config : {bart_config}")
print('-------------------------------------------------------------------------------')

# 2nd Method : Using Directly the models configuration method
from transformers import BertConfig

bert_config2 = BertConfig.from_pretrained('bert-base-cased')

# Using selective layers only
from transformers import BertConfig, BertModel
bert_config = BertConfig.from_pretrained('bert-based-cased', num_hidden_layers = 10)   # Num_hidden_layers tells how many layers you wanna use.
bert_model = BertModel(bert_config)   # Loading the new architecture with only 10 layers to the model.

# Saving a model after it's trained or fine tuned
bert_model.save_pretrained("my-bert-model")

# Reloading the saved model
from transformers import BertModel

bert_model = BertModel.from_pretrained('my-bert-model')
