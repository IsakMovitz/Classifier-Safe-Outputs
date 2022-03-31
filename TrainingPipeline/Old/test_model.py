from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# tokenizer = AutoTokenizer.from_pretrained("AI-Nordics/bert-large-swedish-cased")
# model = AutoModelForSequenceClassification.from_pretrained("AI-Nordics/bert-large-swedish-cased")

# classifier = pipeline(task= 'text-classification',        
#                       model= model,
#                       tokenizer = tokenizer)

# results = classifier(["Hej vad kul att se dig", "Jag hatar dig jävla råtta"])
# for result in results:
#     print(f"label: {result['label']}, with score: {round(result['score'], 4)}")


#######################################

# Test model is a bert-base-cased finetuned with 10 datapoints of imdb data, from finetuned_bert_trainer_api file
tokenizer = AutoTokenizer.from_pretrained("./Test_model")
model = AutoModelForSequenceClassification.from_pretrained("./Test_model")

#classifier = pipeline(task= 'sentiment-analysis')
classifier = pipeline(task= 'text-classification',        
                      model= model,
                      tokenizer = tokenizer)

results = classifier(["We are very happy to show you the Transformers library.", "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

