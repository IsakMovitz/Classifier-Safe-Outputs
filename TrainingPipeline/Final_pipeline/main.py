from transformers import Trainer
from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import DatasetDict
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, DataCollatorWithPadding
from transformers import Trainer,TrainerCallback
from transformers import AutoModelForSequenceClassification

class CustomTrainer(Trainer):
    pass

def load_tokenize_jsonl(finetune_data,train_test_split,tokenizer):

    dataset = load_dataset('json', data_files= finetune_data)['train']
    dataset = dataset.remove_columns(["id","thread_id","thread","keyword","starting_index","span_length"])
    dataset = dataset.rename_column("TOXIC", "label")

    train_test = dataset.train_test_split(test_size=train_test_split)
    train_test_dataset = DatasetDict({
        'train': train_test['train'],
        'test': train_test['test']
    })

    full_datasets = train_test_dataset

    tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = full_datasets.map(tokenize_function, batched=True)

    ### Creating subsets ### 
    tokenized_datasets = []
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10)) # 1000
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10))   # 1000
    full_train_dataset = tokenized_datasets["train"]
    full_eval_dataset = tokenized_datasets["test"]

    tokenized_datasets.append(small_train_dataset)
    tokenized_datasets.append(small_eval_dataset)
    tokenized_datasets.append(full_train_dataset)
    tokenized_datasets.append(full_eval_dataset)

    return tokenized_datasets

def finetune_model(model,train_data,test_data,tokenizer, save_filename):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

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
        model= model,
        args=training_args,
        train_dataset= train_data,
        eval_dataset= test_data,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Start fine-tuning 
    train_result = trainer.train()

    # Log and save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_model(save_filename)
    tokenizer.save_pretrained(save_filename)

def main():

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

    ### Dataset ###
    tokenized_datasets = load_tokenize_jsonl('./FINETUNE_DATASET.jsonl', 0.2 , tokenizer)
    small_train = tokenized_datasets[0]
    small_test = tokenized_datasets[1]

    ### Model training ###
    finetune_model(model,small_train,small_test,tokenizer,"./Finetuned_Model/")
    
    ### Model evaluation ###


if __name__ == '__main__':
    main()