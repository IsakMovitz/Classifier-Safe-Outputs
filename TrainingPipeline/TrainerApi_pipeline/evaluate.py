
import numpy as np
from datasets import load_metric
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TextClassificationPipeline

tokenizer = AutoTokenizer.from_pretrained("TrainerApiModel/")
finetuned_model = AutoModelForSequenceClassification.from_pretrained("TrainerApiModel/")


# metric = load_metric("accuracy")
# # metric = load_metric("recall")
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)


# metrics = trainer.evaluate()
# trainer.log_metrics("eval", metrics)

# predictions, labels, metrics = trainer.predict(test_ds)
# # predictions = np.argmax(predictions, axis=1)
# trainer.log_metrics("test", metrics)



# Testing single strings

# METHOD 1
# classifier = pipeline(task= 'text-classification',        
#                       model= finetuned_model,
#                       tokenizer = tokenizer)
# results = classifier(["This movie is shit.", "We hope you don't hate it."])
# print(results)

# METHOD 2
# pipe = TextClassificationPipeline(model=finetuned_model, tokenizer=tokenizer, return_all_scores=True)
# answer = pipe("this is as string")
# print(answer)

