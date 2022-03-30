from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np
from datasets import load_metric

# Source: https://huggingface.co/docs/transformers/training
# Goal: Classify whether movie reviews in the imdb-dataset are positive or negative.

### Dataset ###
raw_datasets = load_dataset("imdb")

### Pre process data, padding with max_length = 512 ### 
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

### Creating subsets ### 
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10)) # 1000
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10))   # 1000
full_train_dataset = tokenized_datasets["train"]
full_eval_dataset = tokenized_datasets["test"]

### Fine-tune training, using the Trainer API ### 
finetuned_model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
training_args = TrainingArguments("test_trainer")
#training_args = TrainingArguments("test_trainer", evaluation_strategy="epoch")

trainer = Trainer(model= finetuned_model, args=training_args, train_dataset=small_train_dataset, eval_dataset=small_eval_dataset)

# Start fine-tuning# 
trainer.train()

## Save model ##
trainer.save_model("./Models/")
tokenizer.save_pretrained("./Models/")

### Evaluation of the final model, evaluated based on accuracy ###

# metric = load_metric("accuracy")
# #metric = load_metric("glue","mrpc")

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

# new_trainer = Trainer(
#     model=finetuned_model,
#     args=training_args,
#     train_dataset=small_train_dataset,
#     eval_dataset=small_eval_dataset,
#     compute_metrics=compute_metrics,
# )

# # Call evaluation # 
# evaluation_dict = trainer.evaluate()
# print(evaluation_dict)


# ### Prediction ### 

# predictions = trainer.predict(tokenized_datasets["validation"]) # From video ,  Key Error for "validation", that feature does not exist

# preds = np.argmax(predictions.predictions,axis = -1)
# metric.compute(predictions = preds, references= predictions.label_ids)