from functions import *
from transformers import EarlyStoppingCallback

def main():

    # Parameters
    create_seed(42)         # 42, 30 , 20
    pretrained_model = "KB/bert-base-swedish-cased"              # "KB/bert-base-swedish-cased" , "AI-Nordics/bert-large-swedish-cased"
    finetune_dataset = "./FINETUNE_DATASET.jsonl"               # ./MERGED_SHUFFLE_15_20.jsonl, ./FINETUNE_DATASET.jsonl
    run_nr = 10
    model_name = "KB"

    run_name = "Run" + str(run_nr) + "_" + model_name
    final_model_dir = "Local/" + run_name + "/" + model_name + "_Model/"
    train_test_split = 0.3
    test_valid_split = 0.5

    # Training parameters
    training_args = TrainingArguments(
        #f"training_with_callbacks",
        output_dir="Local/" + run_name + "/" + model_name + "_train_output",
        overwrite_output_dir=True,
        logging_strategy= "steps",
        logging_steps =1,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy= "steps",
        report_to="wandb",
        lr_scheduler_type="linear",
        disable_tqdm=True,

        # metric_for_best_model='f1',
        # load_best_model_at_end=True,
        # save_total_limit=5

    )

    # lr_scheduler_type:
    # LINEAR="linear"
    # COSINE = "cosine"
    # COSINE_WITH_RESTARTS = "cosine_with_restarts"
    # POLYNOMIAL = "polynomial"
    # CONSTANT = "constant"
    # CONSTANT_WITH_WARMUP = "constant_with_warmup"
    # warmup_steps=50

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