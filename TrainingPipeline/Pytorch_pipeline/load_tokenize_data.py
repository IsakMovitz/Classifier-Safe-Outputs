from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import AutoModel, AutoTokenizer
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

# train_dataset = full_datasets['train'][0]
# print(train_dataset['text'])
# print(train_dataset['label'])


### Tokenizing based on pretrained model ### 

# BertTokenizer, could use this one as well?
#tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")        # bert-base-cased,KB/bert-base-swedish-cased, AI-Nordics/bert-large-swedish-cased
                                                                                        # https://kb-labb.github.io/posts/2022-03-16-evaluating-swedish-language-models/
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer), Maybe default one is fine?

# Swedish models
#tokenizer = AutoTokenizer.from_pretrained("AI-Nordics/bert-large-swedish-cased")
# model = AutoModelForMaskedLM.from_pretrained("AI-Nordics/bert-large-swedish-cased")
# KB/bert-base-swedish-cased

tokenizer = AutoTokenizer.from_pretrained('KBLab/bert-base-swedish-cased-new')

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length = 20)

tokenized_datasets = full_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])            # Why do we remove this? We remove columns we don't need anymore
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")    # Why rename this?  Since this is required format
tokenized_datasets.set_format("torch")

### Creating subsets ### 
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10)) 
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10))  
full_train_dataset = tokenized_datasets["train"]
full_eval_dataset = tokenized_datasets["test"]

print(full_train_dataset)