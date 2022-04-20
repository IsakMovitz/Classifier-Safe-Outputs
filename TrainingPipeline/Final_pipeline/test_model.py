from functions import *


# Model 3 looks as good according to the test set but seems to perform better when testing strings?


def test_model():


    finetuned_model = AutoModelForSequenceClassification.from_pretrained("./Local/Test6_KB_extended/KB_extended_Model")
    tokenizer = AutoTokenizer.from_pretrained("./Local/Test6_KB_extended/KB_extended_Model")
    result = evaluate_string("Jag gillar alla mina snälla och roliga vänner som är judar och också invandrare neger", finetuned_model, tokenizer)
    print(result)


    # valid = load_data("./Local/VALID_150.jsonl")
    # test = load_data("./Local/TEST_150.jsonl")
    # tokenized_valid = tokenize_data(valid, tokenizer)
    # tokenized_test = tokenize_data(test, tokenizer)
    #
    #
    # evaluate_model(finetuned_model, tokenizer, tokenized_valid, "Local/Local_test_1/")


if __name__ == '__main__':
    test_model()





