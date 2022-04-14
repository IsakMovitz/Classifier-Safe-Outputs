from functions import *
from datasets import load_from_disk

def main():

    # Parameters
    create_seed(20)         # 42, 30 , 20
    pretrained_model = "KB/bert-base-swedish-cased"              # "KB/bert-base-swedish-cased" , "AI-Nordics/bert-large-swedish-cased"
    finetune_dataset = "./FINETUNE_DATASET.jsonl"               # ./MERGED_SHUFFLE_15_20.jsonl, ./FINETUNE_DATASET.jsonl
    run_nr = 3
    model_name = "KB"

    run_name = "Test" + str(run_nr) + "_" + model_name
    final_model_dir = "Local/" + run_name + "/" + model_name + "_Model/"
    train_test_split = 0.3
    test_valid_split = 0.5

    # Training parameters
    training_args = TrainingArguments(
        output_dir="Local/" + run_name + "/" + model_name + "_train_output",
        overwrite_output_dir=True,
        logging_strategy= "steps",
        logging_steps =1,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        evaluation_strategy= "steps",
        report_to="wandb",
        lr_scheduler_type="linear",
        disable_tqdm=True
    )

    # Model and tokenizer
    model, tokenizer, device = load_model_tokenizer_device(pretrained_model)
    model.to(device)

    # 700, 150,150 split
    train = load_from_disk("./Local/datasets/train/")
    valid = load_from_disk("./Local/datasets/valid/")
    test = load_from_disk("./Local/datasets/test/")
    tokenized_train = tokenize_data(train, tokenizer)
    tokenized_valid = tokenize_data(valid, tokenizer)
    tokenized_test = tokenize_data(test, tokenizer)

    # Model training
    train_model(model,final_model_dir,training_args,tokenized_train,tokenized_valid,tokenizer)
    log_args(training_args.output_dir,training_args,pretrained_model,finetune_dataset,train_test_split, test_valid_split)

    # Model evaluation
    finetuned_model = AutoModelForSequenceClassification.from_pretrained(final_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(final_model_dir)

    evaluate_model(finetuned_model, tokenizer, tokenized_test, "Local/" + run_name + "/" + model_name + "_eval_output")

    #evaluate_string("En exempel mening med toxisk text",finetuned_model,tokenizer)

if __name__ == '__main__':
    main()