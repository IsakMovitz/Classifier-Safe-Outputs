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

import numpy as np
from datasets import load_metric
from transformers import pipeline

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

def load_model_tokenizer_device(pretrained_model):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=2)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    return model,tokenizer,device


def finetune_model(model,train_data,test_data,tokenizer, model_filename,results_filename):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir= results_filename,
        # evaluation_strategy= "epoch",   # Evaluate after every epoch  
        logging_strategy= "steps",       
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=20,
        logging_dir='./logs',
        #report_to="wandb"
        #evaluation_strategy="steps"
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

    print(trainer.state.log_history)

    # # Log and save training metrics
    # metrics = train_result.metrics
    # trainer.log_metrics("train", metrics)
    # trainer.save_metrics("train", metrics)
    # trainer.save_model(model_filename)
    # tokenizer.save_pretrained(model_filename)

def train_model():

    pretrained_model = "bert-base-cased"
    model, tokenizer, device = load_model_tokenizer_device(pretrained_model)
    model.to(device)

    full_datasets = load_split_data("./FINETUNE_DATASET.jsonl")

    ### Dataset ###
    tokenized_datasets = tokenize_data(full_datasets, tokenizer)
    small_train = tokenized_datasets[0]
    small_test = tokenized_datasets[1]

    print(small_train)
    
    ### Model training ###
    finetune_model(model,small_train,small_test,tokenizer,"./bert_basecase_test_Model/","./bert_basecase_test_results/")
    
def evaluate_model(model_filename,stats_output_dir):
    ### Load finetuned model from local ###
    tokenizer = AutoTokenizer.from_pretrained(model_filename)
    finetuned_model = AutoModelForSequenceClassification.from_pretrained(model_filename)


    full_datasets = load_split_data("./FINETUNE_DATASET.jsonl")

    ### Dataset ###
    tokenized_datasets = tokenize_data(full_datasets, tokenizer)
    small_test = tokenized_datasets[1]

    ### Evaluate model on test dataset ###
    def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)

            recall_metric = load_metric('recall')
            precision_metric = load_metric('precision')
            accuracy_metric = load_metric('accuracy')
            f1_metric = load_metric('f1')
            #glue_metric = load_metric("glue", "mrpc")

            recall = recall_metric.compute(predictions=predictions, references=labels)
            precision = precision_metric.compute(predictions=predictions, references=labels)
            accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
            f1 = f1_metric.compute(predictions=predictions, references=labels)

            return {"recall": recall,"precision": precision, "accuracy":accuracy, "f1": f1}

    training_args = TrainingArguments(
        output_dir= stats_output_dir
    )

    test_trainer = Trainer(
        model= finetuned_model,
        args=training_args,
        eval_dataset= small_test,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    def log_metrics(metrics):

        for key,value in metrics.items():
            print(key + " = " + str(value) )
        
    metrics = test_trainer.evaluate()

    log_metrics(metrics)

    ### Testing single strings ###

    def evaluate_string(string):
        classifier = pipeline(task= 'text-classification',       # 'sentiment-analysis'    
                        model= finetuned_model,
                        tokenizer = tokenizer)
        result = classifier(string)
        label = result[0]['label']

        if label == 'LABEL_0':
            print("SAFE")
        elif label == 'LABEL_1':
            print('TOXIC')

        return result

def main():
    
    train_model()

    #evaluate_model("bert_basecase_test_Model","bert_basecase_test_results") # "bert_basecase_test_results"
    

if __name__ == '__main__':
    main()