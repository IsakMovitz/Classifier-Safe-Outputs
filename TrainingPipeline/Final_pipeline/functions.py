from transformers import Trainer
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import TrainingArguments, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset
from datasets import DatasetDict
from datasets import load_metric
import torch
import numpy as np
from torch import nn
import random

class CustomTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        self.num_labels = 2
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def create_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def log_args(filepath,training_args,pretrained_model,finetune_dataset,train_test_split, test_valid_split):
    with open( filepath +'/parameters.txt', 'w') as f:
        f.writelines("pretrained_model = " + str(pretrained_model) + "\n")
        f.writelines("finetune_dataset = " + str(finetune_dataset) + "\n")
        f.writelines("train_test_split = " + str(train_test_split) + "\n")
        f.writelines("test_valid_split = " + str(test_valid_split) + "\n")

        f.writelines("--Training arguments--" + "\n")
        f.writelines("loggin strategy = " + str(training_args.logging_strategy)+ "\n")
        f.writelines("logging_steps = " + str(training_args.logging_steps)+ "\n")
        f.writelines("learning_rate = " + str(training_args.learning_rate)+ "\n")
        f.writelines("per_device_train_batch_size = " + str(training_args.per_device_train_batch_size)+ "\n")
        f.writelines("per_device_eval_batch_size = " + str(training_args.per_device_eval_batch_size)+ "\n")
        f.writelines("num_train_epochs = " + str(training_args.num_train_epochs)+ "\n")
        f.writelines("weight_decay = " + str(training_args.weight_decay)+ "\n")
        f.writelines("evaluation_strategy = " + str(training_args.evaluation_strategy) + "\n")
        f.writelines("report_to = " + str(training_args.report_to) + "\n")
        f.writelines("lr_scheduler_type== " + str(training_args.lr_scheduler_type))

def load_split_data(jsonl_file,train_test_split, test_valid_split):

    # Code for train,test,valid split
    dataset = load_dataset('json', data_files=jsonl_file)['train']
    dataset = dataset.remove_columns(["id","thread_id","thread","keyword","starting_index","span_length"])
    dataset = dataset.rename_column("TOXIC", "label")

    train_testvalid = dataset.train_test_split(test_size=train_test_split)
    test_valid = train_testvalid['test'].train_test_split(test_size=test_valid_split)
    train_test_valid_dataset = DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'valid': test_valid['train']})


    full_datasets = train_test_valid_dataset

    return full_datasets

def tokenize_data(dataset,tokenizer):

    def tokenize_function(examples):
        return tokenizer(examples["text"])  # max_length=20, padding="max_length", max_length=128,truncation=True, padding="max_length", max_length = 22,

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(['text'])


    return tokenized_dataset

def load_model_tokenizer_device(pretrained_model):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=2)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    return model,tokenizer,device

def build_trainer(model,train_data,valid_data,training_args, tokenizer):
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = CustomTrainer(          # Trainer, CustomTrainer
        model= model,
        args=training_args,
        train_dataset= train_data,
        eval_dataset= valid_data,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    return trainer

def train_model(pretrained_model,final_model_dir,training_args,full_train,full_valid,tokenizer):

    # Initializing model 
    trainer = build_trainer(pretrained_model,full_train,full_valid,training_args, tokenizer)

    # Finetuning 
    train_results = trainer.train()

    # Log and save training metrics
    metrics = train_results.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)


def evaluate_model(finetuned_model,tokenizer,test_data,stats_output_dir):

    # Evaluate model on test dataset
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred     # logits
        predictions = np.argmax(predictions, axis=1)

        recall_metric = load_metric('recall')
        precision_metric = load_metric('precision')
        accuracy_metric = load_metric('accuracy')
        f1_metric = load_metric('f1')

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
        eval_dataset= test_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    def log_metrics(metrics):

        for key,value in metrics.items():
            print(key + " = " + str(value) )
        
    metrics = test_trainer.evaluate()
    test_trainer.save_metrics("eval",metrics)
    log_metrics(metrics)

def evaluate_string(string,finetuned_model,tokenizer):
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



