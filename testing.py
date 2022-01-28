
### Testing hugginface library ### 
from transformers import pipeline

# distilbert-base-uncased-finetuned-sst-2-english CLASSIFIER, classifying text as positive / negative # 
classifier = pipeline("sentiment-analysis")
results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")


### Testing PyTorch ### 
import torch

mat1 = torch.randn(2, 3)
mat2 = torch.randn(3, 3)
matmult = torch.mm(mat1, mat2)
print(matmult)

print(torch.cuda.is_available())