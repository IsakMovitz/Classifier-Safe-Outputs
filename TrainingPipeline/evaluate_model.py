from datasets import load_metric
import torch
from torch.utils.data import DataLoader

from load_tokenize_data import * 

# Either this
# finetuned_model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
# finetuned_model.load_state_dict(torch.load("SAVED_MODEL.pt"))
# finetuned_model.eval()

# Or this
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

finetuned_model = torch.load("SAVED_MODEL.pkl")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
finetuned_model.to(device)

metric = load_metric("accuracy")

finetuned_model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = finetuned_model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(metric.compute())