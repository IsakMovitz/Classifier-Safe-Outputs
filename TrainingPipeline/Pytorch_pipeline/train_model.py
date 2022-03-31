from transformers import AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from transformers import DataCollatorWithPadding
import numpy as np

from load_tokenize_data import *

# Finetuning works with the ber-base-cased:
# AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
# How to integrate the Swedish models?

### Training model ###
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8) # full_train_dataset
#finetuned_model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

# How would this work for the Swedish models?
#finetuned_model = AutoModelForMaskedLM.from_pretrained("AI-Nordics/bert-large-swedish-cased")
#finetuned_model = AutoModel.from_pretrained('KB/bert-base-swedish-cased')

# KBLab/bert-base-swedish-cased
finetuned_model = AutoModelForSequenceClassification.from_pretrained('KBLab/bert-base-swedish-cased-new', num_labels=2)

optimizer = AdamW(finetuned_model.parameters(), lr=5e-5,no_deprecation_warning=True) # Maybe try other optimizers?

# What is our loss criterion?
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)

# Scheduler used for adjusting learning rate, now probably linear
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
finetuned_model.to(device)

progress_bar = tqdm(range(num_training_steps))

finetuned_model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        #print({k: v.to(device).shape for k, v in batch.items()})
        outputs = finetuned_model(**batch)
        #print(outputs.loss,outputs.logits.shape)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# Either this 
#torch.save(finetuned_model.state_dict(), "SAVED_MODEL.pkl")

# Or this
torch.save(finetuned_model, "OTHER_10_MODEL.pkl")
