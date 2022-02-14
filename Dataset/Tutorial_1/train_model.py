
from imports import *

### 3. Fine-tune the pre-trained model ###
## Load from rubrix ## 

rb_df = rb.load(name='example-dataset')       # Pandas dataframe , query="status:Validated", query="status:Default"
#head = rb_df.head()
print(rb_df)
#print(type(rb_df))

## Prepare training and test datasets ## 

rb_df['text'] = rb_df.inputs.transform(lambda r: r['text'])
rb_df['labels'] = rb_df.annotation

# Make to a list with tex and labels
ds = rb_df[['text', 'labels']].to_dict(orient='list')

# Make into dataser arrow format for training
train_ds = Dataset.from_dict(
    ds,
    features=Features({
        "text": Value("string"),
        "labels": ClassLabel(names=list(rb_df.labels.unique()))
    })
)

# Split into test and train
train_ds = train_ds.train_test_split(test_size=0.2) ; train_ds

#print(train_ds['train'])

# Tokenize 
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_dataset = train_ds['train'].map(tokenize_function, batched=True).shuffle(seed=42)
eval_dataset = train_ds['test'].map(tokenize_function, batched=True).shuffle(seed=42)

#print(train_dataset['attention_mask'])

## Train model ## 
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

import numpy as np
from transformers import Trainer
from datasets import load_metric
from transformers import TrainingArguments

training_args = TrainingArguments(
    "distilbert-base-uncased-sentiment-banking",
    evaluation_strategy="epoch",
    logging_steps=30
)

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()