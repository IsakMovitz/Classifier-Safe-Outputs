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

### Saving the model to pickle file ###
# torch.save({
#             'epoch': epoch,
#             'model_state_dict': finetuned_model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss,
#             }, "SAVED.pth")

### Dataset ###
dataset = load_dataset('json', data_files='./Final_data/RESHUFFLED_FINAL_20SPAN_KEYWORD_DATASET.jsonl')['train']
# train_test_dataset = dataset.train_test_split(train_size= 0.8, test_size=0.2)
# print(train_test_dataset)

train_testvalid = dataset.train_test_split(test_size=0.1)
# Split the 10% test + valid in half test, half valid
test_valid = train_testvalid['test'].train_test_split(test_size=0.1)
# gather everyone if you want to have a single DatasetDict
train_test_valid_dataset = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})


full_datasets = train_test_valid_dataset

### Pre process data, padding with max_length = 512 ### 

## Tokenizing based on pretrained model

tokenizer = AutoTokenizer.from_pretrained("AI-Nordics/bert-large-swedish-cased")        # KB/bert-base-swedish-cased, AI-Nordics/bert-large-swedish-cased
                                                                    # https://kb-labb.github.io/posts/2022-03-16-evaluating-swedish-language-models/
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = full_datasets.map(tokenize_function, batched=True)

### Creating subsets ### 
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10)) # 1000
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10))   # 1000
full_train_dataset = tokenized_datasets["train"]
full_eval_dataset = tokenized_datasets["test"]

### Fine-tune training, using the Trainer API ### 
finetuned_model = AutoModelForSequenceClassification.from_pretrained("AI-Nordics/bert-large-swedish-cased", num_labels=2)
training_args = TrainingArguments("test_trainer")
#training_args = TrainingArguments("test_trainer", evaluation_strategy="epoch")

trainer = Trainer(model= finetuned_model, args=training_args, train_dataset=small_train_dataset, eval_dataset=small_eval_dataset)

# Start fine-tuning# 
trainer.train()

## Save model ##
trainer.save_model("./Swe_finetuned_model/")
tokenizer.save_pretrained("./Swe_finetuned_model/")

### Evaluation of the final model, evaluated based on accuracy ###

metric = load_metric("accuracy")
#metric = load_metric("glue","mrpc")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

new_trainer = Trainer(
    model=finetuned_model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

# Call evaluation # 
evaluation_dict = trainer.evaluate()
print(evaluation_dict)
