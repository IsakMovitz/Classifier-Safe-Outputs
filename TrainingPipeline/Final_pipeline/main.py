from functions import *

def main():

    # Parameters
    create_seed(42)
    pretrained_model = "KB/bert-base-swedish-cased"  # "KB/bert-base-swedish-cased"
    finetune_dataset = "./FINETUNE_DATASET.jsonl"
    run_name = "Run6"
    model_name = "KB"
    final_model_dir = "Local/" + run_name + "/" + model_name + "_Model/"
    train_test_split = 0.3
    test_valid_split = 0.5

    # Training parameters
    training_args = TrainingArguments(
        output_dir="Local/" + run_name + "/" + model_name + "_train_output",
        overwrite_output_dir=True,
        logging_strategy= "steps",
        logging_steps =1,
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy= "epoch",
        report_to="wandb",
    )

    # Model and tokenizer
    model, tokenizer, device = load_model_tokenizer_device(pretrained_model)
    model.to(device)

    # Finetuning dataset
    full_datasets = load_split_data(finetune_dataset, train_test_split, test_valid_split)
    tokenized_datasets = tokenize_data(full_datasets, tokenizer)
    full_train = tokenized_datasets[0]
    full_valid = tokenized_datasets[1]
    full_test = tokenized_datasets[2]

    # Model training
    train_model(model,final_model_dir,training_args,full_train,full_valid,tokenizer)
    log_args(training_args.output_dir,training_args,pretrained_model,finetune_dataset,train_test_split, test_valid_split)

    # Model evaluation
    finetuned_model = AutoModelForSequenceClassification.from_pretrained(final_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(final_model_dir)

    evaluate_model(finetuned_model, tokenizer, full_test, "Local/" + run_name + "/" + model_name + "_eval_output")

    #evaluate_string("En exempel mening med toxisk text",finetuned_model,tokenizer)

if __name__ == '__main__':
    main()