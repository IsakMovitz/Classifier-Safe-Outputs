
from datasets import load_dataset
from transformers import pipeline
import rubrix as rb
from datasets import Dataset, Features, Value, ClassLabel
from transformers import AutoTokenizer

#### TUTORIAL: Label your data to fine-tune a classifier with Hugging Face ####

### 1. Run the pre-trained model over the dataset and log the predictions ###
## Dataset ## 
banking_ds = load_dataset("banking77")
to_label1, to_label2 = banking_ds['train'].train_test_split(test_size=0.5, seed=42).values()
# print(to_label1[0])
# print(to_label1[1])

## Classifier ## 
sentiment_classifier = pipeline(
    model="distilbert-base-uncased-finetuned-sst-2-english",
    task="sentiment-analysis",
    return_all_scores=True,
)
#example_data = to_label1[3]['text']
#print(example_data)
#print(sentiment_classifier(example_data))

## Prediction ## 
def predict(examples):
    return {"predictions": sentiment_classifier(examples['text'], truncation=True)}

# add.select(range(10)) before map if you just want to test this quickly with 10 examples
to_label1 = to_label1.select(range(10)).map(predict, batched=True, batch_size=4)
#print(to_label1['text'])
#print(to_label1['label'])
#print(to_label1['predictions'])

## Record to rubrix ## 
records = []
for example in to_label1.shuffle(): # Looping over predictions  
    record = rb.TextClassificationRecord(
        inputs=example["text"],
        metadata={'category': example['label']}, # log the intents for exploration of specific intents
        prediction=[(pred['label'], pred['score']) for pred in example['predictions']],
        prediction_agent="distilbert-base-uncased-finetuned-sst-2-english"
    )
    records.append(record)

rb.log(name='labeling_with_pretrained_10', records=records)

### 2. Explore and label data with the pretrained model ###
    # Done in the rubrix web interface

### 3. Fine-tune the pre-trained model ###

rb_df = rb.load(name='labeling_with_pretrained_10', query="status:Validated")       # Pandas dataframe , query="status:Validated", query="status:Default"

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


### 4. Testing the fine-tuned model ###

finetuned_sentiment_classifier = pipeline(
    model=model.to("cpu"),
    tokenizer=tokenizer,
    task="sentiment-analysis",
    return_all_scores=True
)

finetuned_sentiment_classifier(
    'I need to deposit my virtual card, how do i do that.'
), sentiment_classifier(
    'I need to deposit my virtual card, how do i do that.'
)


### 5. Run our fine-tuned model over the dataset and log the predictions ###

rb_df = rb.load(name='labeling_with_pretrained', query="status:Default")
rb_df['text'] = rb_df.inputs.transform(lambda r: r['text'])
ds = Dataset.from_pandas(rb_df[['text']])

def predict(examples):
    return {"predictions": finetuned_sentiment_classifier(examples['text'])}

ds = ds.map(predict, batched=True, batch_size=8)

records = []
for example in ds.shuffle():
    record = rb.TextClassificationRecord(
        inputs=example["text"],
        prediction=[(pred['label'], pred['score']) for pred in example['predictions']],
        prediction_agent="distilbert-base-uncased-banking77-sentiment"
    )
    records.append(record)

rb.log(name='labeling_with_finetuned', records=records)

### 6. Explore and label data with the fine-tuned model ###
    # Done in the rubrix web interface

### 7. Fine-tuning with the extended training dataset ###

# Adding labeled examples to our previous training set
def prepare_train_df(dataset_name):
    rb_df = rb.load(name=dataset_name)
    rb_df = rb_df[rb_df.status == "Validated"] ; len(rb_df)
    rb_df['text'] = rb_df.inputs.transform(lambda r: r['text'])
    rb_df['labels'] = rb_df.annotation.transform(lambda r: r[0])
    return rb_df

df = prepare_train_df('labeling_with_finetuned')
train_dataset = train_dataset.remove_columns('__index_level_0__')


for i,r in df.iterrows():
    tokenization = tokenizer(r["text"], padding="max_length", truncation=True)
    train_dataset = train_dataset.add_item({
        "attention_mask": tokenization["attention_mask"],
        "input_ids": tokenization["input_ids"],
        "labels": label2id[r['labels']],
        "text": r['text'],
    })

# Training our sentiment classifier
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
train_ds = train_dataset.shuffle(seed=42)

trainer = Trainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
model.save_pretrained("distilbert-base-uncased-sentiment-banking")