from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import AutoModel,AutoTokenizer
from transformers import TrainingArguments, DataCollatorWithPadding
from transformers import Trainer
from transformers import AutoModelForSequenceClassification
import numpy as np
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForTokenClassification

def load_tokenize_finetune_dataset():
    ### Dataset ###
    dataset = load_dataset('json', data_files='./Final_data/RESHUFFLED_FINAL_20SPAN_KEYWORD_DATASET.jsonl')['train']
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
    small_test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10))   # 1000
    full_train_dataset = tokenized_datasets["train"]
    full_eval_dataset = tokenized_datasets["test"]

    return tokenizer, small_train_dataset, small_test_dataset

def train_save_model(tokenizer,train_dataset,test_dataset):
    finetuned_model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy= "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model= finetuned_model,
        args=training_args,
        train_dataset= train_dataset,
        eval_dataset= test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Start fine-tuning# 
    trainer.train()

    ## Save model ##
    # trainer.save_model("./new_Model/")
    # tokenizer.save_pretrained("./new_Model/")

    return training_args,trainer


def evaluate_model():

    model = AutoModelForSequenceClassification.from_pretrained('./new_Model')
    tokenizer = AutoTokenizer.from_pretrained('./new_Model')

    metric = load_metric("accuracy")

    # Call evaluation # 
    evaluation_dict = model.evaluate()
    print(evaluation_dict)

    # predictions = trainer.predict(test_dataset) # From video ,  Key Error for "validation", that feature does not exist

    # preds = np.argmax(predictions.predictions,axis = -1)
    # metric.compute(predictions = preds, references= predictions.label_ids)

def main():

    # tokenizer, small_train_dataset,small_test_dataset = load_tokenize_finetune_dataset()

    # print(small_train_dataset)
    # training_args,trainer = train_save_model(tokenizer,small_train_dataset,small_test_dataset)

    evaluate_model()

main()