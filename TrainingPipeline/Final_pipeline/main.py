from functions import *
from datasets import load_from_disk,concatenate_datasets

def main():

    # Parameters
    create_seed(20)         # 42, 30 , 20
    pretrained_model = "AI-Nordics/bert-large-swedish-cased"              # "KB/bert-base-swedish-cased" , "AI-Nordics/bert-large-swedish-cased"
    run_nr = 20
    model_name = "AI-Nord_extended"

    run_name = "Test" + str(run_nr) + "_" + model_name
    final_model_dir = "Local/" + run_name + "/" + model_name + "_Model/"

    # Training parameters
    training_args = TrainingArguments(
        output_dir="Local/" + run_name + "/" + model_name + "_train_output",
        overwrite_output_dir=True,
        logging_strategy= "steps",
        logging_steps =1,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy= "steps",
        report_to="wandb",
        lr_scheduler_type="linear",
        disable_tqdm=True
    )

    # Model and tokenizer
    model, tokenizer, device = load_model_tokenizer_device(pretrained_model)
    model.to(device)

    # Load data
    train = load_data("./Local/TRAIN_700.jsonl")
    valid = load_data("./Local/VALID_150.jsonl")
    test = load_data("./Local/TEST_150.jsonl")
    tokenized_train = tokenize_data(train, tokenizer)
    tokenized_valid = tokenize_data(valid, tokenizer)
    tokenized_test = tokenize_data(test, tokenizer)

    extended_train = load_data("./Local/EXTENDED_TRAIN_1200.jsonl")
    tokenized_extended_train = tokenize_data(extended_train, tokenizer)

    # print(test[0])
    # token = tokenized_test[0]['input_ids']
    # print(token)
    # decoded = tokenizer.decode(tokenized_test[0]['input_ids'])
    # print(decoded)

    # # Model training
    train_model(model,final_model_dir,training_args,tokenized_extended_train,tokenized_valid,tokenizer)
    log_args(training_args.output_dir,training_args,pretrained_model,"TRAIN_700",0.3, 0.5)

    # Model evaluation
    finetuned_model = AutoModelForSequenceClassification.from_pretrained(final_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(final_model_dir)

    evaluate_model(finetuned_model, tokenizer, tokenized_test, "Local/" + run_name + "/" + model_name + "_eval_output")

    #evaluate_string("En exempel mening med toxisk text",finetuned_model,tokenizer)

if __name__ == '__main__':
    main()