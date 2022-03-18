from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from datasets import load_metric
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm

# Source: https://huggingface.co/docs/transformers/training
# Goal: Classify whether movie reviews in the imdb-dataset are positive or negative.

### Dataset ###
raw_datasets = load_dataset("imdb")

# ### Pre process data, padding with max_length = 512 ### 

# Tokenizing based on pretrained model
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

full_train_dataset = tokenized_datasets["train"]
full_eval_dataset = tokenized_datasets["test"]

# ### Fine-tune training, using native PyTorch ### 

# # Free up memory?  #
# # del finetuned_model
# # del trainer
# torch.cuda.empty_cache()

tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10)) # 1000
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10))   # 1000     

print(small_train_dataset['labels'])

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

finetuned_model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
optimizer = AdamW(finetuned_model.parameters(), lr=5e-5)

num_epochs = 3
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


### Evaluation of the final model, evaluated based on accuracy ###

metric = load_metric("accuracy")
finetuned_model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = finetuned_model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(metric.compute())