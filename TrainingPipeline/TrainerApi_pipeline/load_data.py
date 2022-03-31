from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import AutoModel,AutoTokenizer
# A split for train, test, eval can be found in "flashback_finetune.py" file in Old

### Dataset ###
dataset = load_dataset('json', data_files='./RESHUFFLED_FINAL_20SPAN_KEYWORD_DATASET.jsonl')['train']
dataset = dataset.remove_columns(["id","thread_id","thread","keyword","starting_index","span_length"])
dataset = dataset.rename_column("TOXIC", "label")

train_test = dataset.train_test_split(test_size=0.2)
train_test_dataset = DatasetDict({
    'train': train_test['train'],
    'test': train_test['test']
})

full_datasets = train_test_dataset

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = full_datasets.map(tokenize_function, batched=True)

### Creating subsets ### 
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10)) # 1000
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10))   # 1000
full_train_dataset = tokenized_datasets["train"]
full_eval_dataset = tokenized_datasets["test"]

print(small_train_dataset)
