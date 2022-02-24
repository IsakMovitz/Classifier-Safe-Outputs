from curses import raw
from datasets import load_dataset

# Source: https://huggingface.co/docs/datasets/installation.html
# https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset

### IMDB ###

# The dataset is a datasets.Dataset object
imdb_datasets = load_dataset("imdb")

train = imdb_datasets["train"]
test = imdb_datasets["test"]
#print(imdb_datasets)

# Acessing the data 
print(imdb_datasets['train'][0])
print(train[0])

print("########################")

### GLUE , access a configuration (subset) within a dataset ###
dataset = load_dataset('glue', 'mrpc', split='train')

#print(dataset.info.citation)
