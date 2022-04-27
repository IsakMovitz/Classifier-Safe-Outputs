from functions import *
import json

# Model 3 looks as good according to the test set but seems to perform better when testing strings?
# Maybe finding some edge case examples could be good for the study

def find_wrong_classified(input_filename, output_filename):
    cleaned_data = []
    finetuned_model = AutoModelForSequenceClassification.from_pretrained("./Local/Test6_KB_extended/KB_extended_Model")
    tokenizer = AutoTokenizer.from_pretrained("./Local/Test6_KB_extended/KB_extended_Model")


    #Open annotation file from Prodigy:
    with open(input_filename, 'r',encoding='utf-8') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        line_dict = json.loads(json_str)
        text_sample = line_dict['text']
        toxic = line_dict['TOXIC']

        result = evaluate_string(text_sample,finetuned_model, tokenizer)
        if toxic != result:
            cleaned_data.append(line_dict)

    #Save cleaned back to jsonl format:
    with open(output_filename, 'w', encoding='utf-8') as f:
            for item in cleaned_data:
                if item != cleaned_data[-1]:
                    f.write(json.dumps(item,ensure_ascii=False) + "\n")
                else:
                    f.write(json.dumps(item,ensure_ascii=False))

def test_model():

    finetuned_model = AutoModelForSequenceClassification.from_pretrained("./Local/Test6_KB_extended/KB_extended_Model")
    tokenizer = AutoTokenizer.from_pretrained("./Local/Test6_KB_extended/KB_extended_Model")
    print("-----")
    result = evaluate_string("Vad fan vill du vi ska diskutera Nog för att jag hatar feminister och vill att de avlider snabbt men", finetuned_model, tokenizer)
    print("-----")
    print(result)


    # valid = load_data("./Local/VALID_150.jsonl")
    # test = load_data("./Local/TEST_150.jsonl")
    # tokenized_valid = tokenize_data(valid, tokenizer)
    # tokenized_test = tokenize_data(test, tokenizer)
    #
    #
    # evaluate_model(finetuned_model, tokenizer, tokenized_valid, "Local/Local_test_1/")


if __name__ == '__main__':
    #test_model()
    #find_wrong_classified("./Local/TEST_150.jsonl","./Local/WRONG_CLASSIFIED_KB.jsonl")
    pass




