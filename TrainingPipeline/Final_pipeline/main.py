from sklearn.model_selection import train_test_split
from functions import *

def main():
    
     ### Training parameters ###

    pretrained_model = "KB/bert-base-swedish-cased"                                      # "bert-base-cased", "KB/bert-base-swedish-cased"
    finetune_dataset = "./FINETUNE_DATASET.jsonl"
    final_model_dir = "./KB_test_Model/"
    train_test_split = 0.2

    training_args = TrainingArguments(
        output_dir= "./KB_train_output/",
        overwrite_output_dir=True, 
        logging_strategy= "steps",
        logging_steps = 1,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy= "steps",
        report_to="wandb",
    )

    #train_model(pretrained_model,finetune_dataset,final_model_dir,training_args)

    ### Evaluation of model ###
    finetuned_model = AutoModelForSequenceClassification.from_pretrained("KB_test_Model")
    tokenizer = AutoTokenizer.from_pretrained("KB_test_Model")
    full_datasets = load_split_data("FINETUNE_DATASET.jsonl", train_test_split)
    tokenized_datasets = tokenize_data(full_datasets, tokenizer)
    small_test = tokenized_datasets[1]

    evaluate_model(finetuned_model, tokenizer, small_test, "./KB_eval_output")
    # evaluate_string("En exempel mening med toxisk text",finetuned_model,tokenizer)

if __name__ == '__main__':
    main()