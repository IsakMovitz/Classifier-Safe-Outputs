from datasets import load_metric
import torch
from torch.utils.data import DataLoader
from transformers import TextClassificationPipeline

from load_tokenize_data import * 

# Either this
# finetuned_model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
# finetuned_model.load_state_dict(torch.load("SAVED_MODEL.pt"))
# finetuned_model.eval()

# Or this
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

finetuned_model = torch.load("OTHER_10_MODEL.pkl")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
finetuned_model.to(device)

tokenizer = AutoTokenizer.from_pretrained('AI-Nordics/bert-large-swedish-cased')
pipe = TextClassificationPipeline(model=finetuned_model, tokenizer=tokenizer, return_all_scores=True)
answer = pipe("till sverige är irrelevant vilken idiot hade inte tagit chansen att dra från Mogadishu om möjligheten erbjuds Vad du bör")
print(answer)

# [{'label': 'LABEL_0', 'score': 0.3704254627227783}, {'label': 'LABEL_1', 'score': 0.6295745372772217}]]
# LABEL_0 = 0 = non-toxic    ,  LABEL_1 = 1 = toxic  i think

metric = load_metric("accuracy")
finetuned_model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = finetuned_model(**batch)

    # probability scores for every instance in the batch
    logits = outputs.logits

    print(logits)

    # predictions = torch.argmax(logits, dim=-1)
    # print(predictions)
    
    # metric.add_batch(predictions=predictions, references=batch["labels"])

# print(metric.compute())