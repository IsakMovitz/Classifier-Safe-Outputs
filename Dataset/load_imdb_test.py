from datasets import load_dataset

### IMDB ###
# The dataset is a datasets.Dataset object
imdb_datasets = load_dataset("imdb")

train = imdb_datasets["train"]
test = imdb_datasets["test"]
print(train[0])

