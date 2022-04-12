#from sklearn.model_selection import train_test_split
from functions import *

def log_args(filepath,training_args,pretrained_model,finetune_dataset,train_test_split):
    with open( filepath +'/parameters.txt', 'w') as f:
        f.writelines("pretrained_model = " + str(pretrained_model) + "\n")
        f.writelines("finetune_dataset = " + str(finetune_dataset) + "\n")
        f.writelines("train_test_split = " + str(train_test_split) + "\n")
        f.writelines("--Training arguments--")
        f.writelines("loggin strategy = " + str(training_args.logging_strategy)+ "\n")
        f.writelines("logging_steps = " + str(training_args.logging_steps)+ "\n")
        f.writelines("learning_rate = " + str(training_args.learning_rate)+ "\n")
        f.writelines("per_device_train_batch_size = " + str(training_args.per_device_train_batch_size)+ "\n")
        f.writelines("per_device_eval_batch_size = " + str(training_args.per_device_eval_batch_size)+ "\n")
        f.writelines("num_train_epochs = " + str(training_args.num_train_epochs)+ "\n")
        f.writelines("weight_decay = " + str(training_args.weight_decay)+ "\n")
        f.writelines("evaluation_strategy = " + str(training_args.evaluation_strategy)+ "\n")
        f.writelines("report_to = " + str(training_args.report_to))

def main():
    
     ### Training parameters ###

    # Should save all parameters somewhere 

    pretrained_model = "KB/bert-base-swedish-cased"                                      # "bert-base-cased", "KB/bert-base-swedish-cased"
    finetune_dataset = "./FINETUNE_DATASET.jsonl"
    final_model_dir = "./Run2_KB_Model/"
    train_test_split = 0.2

    training_args = TrainingArguments(
        output_dir= "./Run2_KB_train_output/",
        overwrite_output_dir=True, 
        logging_strategy= "steps",
        logging_steps = 1,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        #weight_decay=0.01,
        evaluation_strategy= "epoch",
        report_to="wandb",
    )

    # train_model(pretrained_model,finetune_dataset,final_model_dir,training_args,train_test_split)
    # log_args(training_args.output_dir,training_args,pretrained_model,finetune_dataset,train_test_split)

    ### Evaluation of model ###
    finetuned_model = AutoModelForSequenceClassification.from_pretrained("KB_test_Model")
    tokenizer = AutoTokenizer.from_pretrained("KB_test_Model")
    full_datasets = load_split_data("FINETUNE_DATASET.jsonl", train_test_split)
    tokenized_datasets = tokenize_data(full_datasets, tokenizer)
    small_test = tokenized_datasets[1]

    #evaluate_model(finetuned_model, tokenizer, small_test, "./KB_eval_output")
    # evaluate_string("En exempel mening med toxisk text",finetuned_model,tokenizer)

if __name__ == '__main__':
    main()