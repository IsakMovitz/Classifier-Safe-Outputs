
from load_data import * 
from transformers import TrainingArguments, DataCollatorWithPadding
from transformers import Trainer
from transformers import AutoModelForSequenceClassification
import numpy as np
from datasets import load_metric

### Fine-tune training, using the Trainer API ### 
finetuned_model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./results",
    # evaluation_strategy= "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
)
def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        metric = load_metric('accuracy')
        return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model= finetuned_model,
    args=training_args,
    train_dataset= small_train_dataset,
    eval_dataset= small_eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Start fine-tuning# 
trainer.train()

## Save model ##
# trainer.save_model("./TrainerApiModel/")
# tokenizer.save_pretrained("./TrainerApiModel/")

metrics = trainer.evaluate()
trainer.log_metrics("eval", metrics)

predictions, labels, metrics = trainer.predict(small_eval_dataset)

print(predictions)
# predictions = np.argmax(predictions, axis=1)
trainer.log_metrics("test", metrics)


preds = np.argmax(predictions.predictions, axis=-1)

metric = load_metric('glue', 'mrpc')
metric.compute(predictions=preds, references=predictions.label_ids)