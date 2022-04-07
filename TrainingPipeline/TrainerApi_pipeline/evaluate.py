
from cmath import log
import numpy as np
from datasets import load_metric
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer
from transformers import TextClassificationPipeline
from load_data import * 
from transformers import TrainingArguments

### Load finetuned model from local ###
tokenizer = AutoTokenizer.from_pretrained("./TrainerApiModel/")
finetuned_model = AutoModelForSequenceClassification.from_pretrained("./TrainerApiModel/")

### Evaluate model on test dataset ###
def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        print(predictions)
        predictions = np.argmax(predictions, axis=1)
        print(predictions)
        recall_metric = load_metric('recall')
        precision_metric = load_metric('precision')
        accuracy_metric = load_metric('accuracy')
        f1_metric = load_metric('f1')
        #glue_metric = load_metric("glue", "mrpc")

        recall = recall_metric.compute(predictions=predictions, references=labels)
        precision = precision_metric.compute(predictions=predictions, references=labels)
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(predictions=predictions, references=labels)

        return {"recall": recall,"precision": precision, "accuracy":accuracy, "f1": f1}

training_args = TrainingArguments(
    output_dir="./results"
)

new_trainer = Trainer(
    model= finetuned_model,
    args=training_args,
    train_dataset= small_train_dataset,
    eval_dataset= small_eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

def log_metrics(metrics):

    for key,value in metrics.items():
        print(key + " = " + str(value) )
    


metrics = new_trainer.evaluate()
print(metrics)
#new_trainer.log_metrics("eval", metrics)
# new_trainer.save_metrics('eval', metrics)

# log_metrics(metrics)



### Testing single strings ###

def evaluate_string(string):
    classifier = pipeline(task= 'text-classification',       # 'sentiment-analysis'    
                      model= finetuned_model,
                      tokenizer = tokenizer)
    result = classifier(string)
    label = result[0]['label']

    if label == 'LABEL_0':
        print("SAFE")
    elif label == 'LABEL_1':
        print('TOXIC')

    return result


# answer = evaluate_string("This movie is shit")
# print(answer)


# METHOD 1
# classifier = pipeline(task= 'sentiment-analysis',       # 'text-classification'    
#                       model= finetuned_model,
#                       tokenizer = tokenizer)
# results = classifier(["This movie is shit.", "We hope you don't hate it."])


# print(results)

# METHOD 2
# pipe = TextClassificationPipeline(model=finetuned_model, tokenizer=tokenizer, return_all_scores=True)
# answer = pipe("this is as string")
# print(answer)
