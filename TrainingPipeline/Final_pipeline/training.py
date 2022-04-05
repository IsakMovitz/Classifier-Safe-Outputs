
from load_data import * 
from transformers import TrainingArguments, DataCollatorWithPadding
from transformers import Trainer,TrainerCallback
from transformers import AutoModelForSequenceClassification
import numpy as np
from datasets import load_metric
from copy import deepcopy

# Fine-tune training, using the Trainer API 
finetuned_model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Does an evaluation of evaluation dataset after every epoch with "evaluation_strategy" = "epoch"
training_args = TrainingArguments(
    output_dir="./results",
    # evaluation_strategy= "epoch",   # Evaluate after every epoch  
    # logging_strategy= "epoch",       
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model= finetuned_model,
    args=training_args,
    train_dataset= small_train_dataset,
    eval_dataset= small_eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Start fine-tuning 
train_result = trainer.train()

# Log and save training metrics
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_model("./TrainerApiModel/")
tokenizer.save_pretrained("./TrainerApiModel/")
