from functions import *

def main():

    # Parameters
    create_seed(42)
    train_test_split = 0.2
    pretrained_model = "AI-Nordics/bert-large-swedish-cased" #  "KB/bert-base-swedish-cased"
    finetune_dataset = "./FINETUNE_DATASET.jsonl"
    run_name = "Run4"
    model_name = "AI_SWE"

    final_model_dir = "Local/" + run_name + "/" + model_name + "_Model/"

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

    full_test = train_model(pretrained_model,finetune_dataset,final_model_dir,training_args,train_test_split)

    log_args(training_args.output_dir,training_args,pretrained_model,finetune_dataset,train_test_split)

    ### Evaluation of model ###
    finetuned_model = AutoModelForSequenceClassification.from_pretrained(final_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(final_model_dir)

    evaluate_model(finetuned_model, tokenizer, full_test, "Local/" + run_name + "/" + model_name + "_eval_output")

    #evaluate_string("En exempel mening med toxisk text",finetuned_model,tokenizer)

if __name__ == '__main__':
    main()