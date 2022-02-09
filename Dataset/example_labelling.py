
### Label your data to fine-tune a classifier with Hugging Face ###
from datasets import load_dataset

banking_ds = load_dataset("banking77")

to_label1, to_label2 = banking_ds['train'].train_test_split(test_size=0.5, seed=42).values()

# print(to_label1[0])
# print(to_label1[1])

from transformers import pipeline

sentiment_classifier = pipeline(
    model="distilbert-base-uncased-finetuned-sst-2-english",
    task="sentiment-analysis",
    return_all_scores=True,
)

example_data = to_label1[3]['text']

print(example_data)
print(sentiment_classifier(example_data))


def predict(examples):
    return {"predictions": sentiment_classifier(examples['text'], truncation=True)}

# add .select(range(10)) before map if you just want to test this quickly with 10 examples
to_label1 = to_label1.map(predict, batched=True, batch_size=4)

import rubrix as rb

records = []
for example in to_label1.shuffle():
    record = rb.TextClassificationRecord(
        inputs=example["text"],
        metadata={'category': example['label']}, # log the intents for exploration of specific intents
        prediction=[(pred['label'], pred['score']) for pred in example['predictions']],
        prediction_agent="distilbert-base-uncased-finetuned-sst-2-english"
    )
    records.append(record)

rb.log(name='labeling_with_pretrained', records=records)