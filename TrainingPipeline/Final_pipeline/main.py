from sklearn.model_selection import train_test_split
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
import torch

class CustomTrainer(Trainer):
    pass

def load_split_data(jsonl_file):
    # Splitting data into train,test,valid

    # train_testvalid = dataset.train_test_split(test_size=0.1)
    # # Split the 10% test + valid in half test, half valid
    # test_valid = train_testvalid['test'].train_test_split(test_size=0.1)
    # # gather everyone if you want to have a single DatasetDict
    # train_test_valid_dataset = DatasetDict({
    #     'train': train_testvalid['train'],
    #     'test': test_valid['test'],
    #     'valid': test_valid['train']})
    # full_datasets = train_test_valid_dataset

    # Splitting data into train,test,valid
    train_test_split = 0.2

    dataset = load_dataset('json', data_files= jsonl_file)['train']
    dataset = dataset.remove_columns(["id","thread_id","thread","keyword","starting_index","span_length"])
    dataset = dataset.rename_column("TOXIC", "label")

    train_test = dataset.train_test_split(test_size=train_test_split)
    train_test_dataset = DatasetDict({
        'train': train_test['train'],
        'test': train_test['test']
    })

    full_datasets = train_test_dataset

    return full_datasets

def tokenize_data(full_datasets,tokenizer):

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = full_datasets.map(tokenize_function, batched=True)

    ### Creating subsets ### 
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10)) # 1000
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10))   # 1000
    full_train_dataset = tokenized_datasets["train"]
    full_eval_dataset = tokenized_datasets["test"]

    tokenized_datasets = []
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
        num_train_epochs=1,
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

def load_model_tokenizer_device(pretrained_model):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=2)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    return model,tokenizer,device

def evaluate_model():
    pass

def main():
    pass
    # pretrained_model = "bert-base-cased"
    # model, tokenizer, device = load_model_tokenizer_device(pretrained_model)
    # model.to(device)

    # full_datasets = load_split_data("./FINETUNE_DATASET.jsonl")

    # ### Dataset ###
    # tokenized_datasets = tokenize_data(full_datasets, tokenizer)
    # small_train = tokenized_datasets[0]
    # small_test = tokenized_datasets[1]

    # print(small_train)
    
    # # ### Model training ###
    # finetune_model(model,small_train,small_test,tokenizer,"./Finetuned_Model/")
    
    ### Model evaluation ###




if __name__ == '__main__':
    main()