from transformers import AutoTokenizer
import torch
from transformers import AutoModelForSequenceClassification

class DocumentEmbedding:
    def __init__(self, model_name):
        model_name = f"models/{model_name}"
        
        if "uncased" in model_name:
            self.lower = True
        else:
            self.lower = False
            
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=3
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}
        
    def predict(self, sentence):
        if self.lower:
            sentence = sentence.lower()
            
        tokenized_sentence = self.tokenizer(sentence, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**tokenized_sentence).logits[0]
        label = logits.argmax(axis=-1)
        return self.id2label[int(label)]
        
