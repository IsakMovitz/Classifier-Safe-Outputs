from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from datasets import load_metric,DatasetDict
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

# Continue with removing unnecessary columns , you only need text, label like with the imdb dataset
# Compare with finetuned_bert_pytorch.py and see what is different

### Dataset ###
dataset = load_dataset('json', data_files='./Final_data/RESHUFFLED_FINAL_20SPAN_KEYWORD_DATASET.jsonl')['train']
dataset = dataset.remove_columns(["id","thread_id","thread","keyword","starting_index","span_length"])
dataset = dataset.rename_column("TOXIC", "label")

# print(type(dataset))
# print(dataset)

train_testvalid = dataset.train_test_split(test_size=0.2)
# Split the 10% test + valid in half test, half valid
train_test_valid_dataset = DatasetDict({
    'train': train_testvalid['train'],
    'test': train_testvalid['test']
})

# test_valid = train_testvalid['test'].train_test_split(test_size=0.1)
# # gather everyone if you want to have a single DatasetDict
# train_test_valid_dataset = DatasetDict({
#     'train': train_testvalid['train'],
#     'test': test_valid['test'],
#     'valid': test_valid['train']})

full_datasets = train_test_valid_dataset

## Tokenizing based on pretrained model

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")        # KB/bert-base-swedish-cased, AI-Nordics/bert-large-swedish-cased
                                                                                        # https://kb-labb.github.io/posts/2022-03-16-evaluating-swedish-language-models/
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = full_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])        # Why do we remove this? 
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
print(tokenized_datasets)

# ### Creating subsets ### 
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10)) # 1000
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10))   # 1000
# full_train_dataset = tokenized_datasets["train"]
# full_eval_dataset = tokenized_datasets["test"]


# ### Training model ###
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

finetuned_model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
optimizer = AdamW(finetuned_model.parameters(), lr=5e-5,no_deprecation_warning=True)

num_epochs = 1 # 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
finetuned_model.to(device)

progress_bar = tqdm(range(num_training_steps))

finetuned_model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
   
        outputs = finetuned_model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


