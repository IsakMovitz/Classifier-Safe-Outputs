from curses import raw
from datasets import load_dataset

# Source: https://huggingface.co/docs/datasets/installation.html

### IMDB ###
imdb_datasets = load_dataset("imdb")
train = imdb_datasets["train"]
test = imdb_datasets["test"]
print(imdb_datasets)

print("########################")

### GLUE ###
dataset = load_dataset('glue', 'mrpc', split='train')

print(dataset.info.citation)
