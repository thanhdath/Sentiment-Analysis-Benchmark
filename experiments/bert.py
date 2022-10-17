import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from transformers import AutoTokenizer
import torch
from datasets import load_dataset
from datasets import Dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='bert-base-uncased')
args = parser.parse_args()

tweet_train_path = "data/combine/kfolds_0/train.csv"
tweet_test_path = "data/combine/kfolds_0/test.csv"
bert_model = args.model
lower = True


def read_text_and_label(file):
    df = pd.read_csv(file)
    data = []

    for _, label, text in df.values:
        label_number = 0
        if label == "negative":
            label_number = 0
        elif label == "neutral":
            label_number = 1
        elif label == "positive":
            label_number = 2
        if lower:
            text = text.lower()
        data.append({"label": label_number, "text": text})
    return data


""" 
80% Training , 20% Testing
"""

train_data = read_text_and_label(tweet_train_path)
test_data = read_text_and_label(tweet_test_path)

train_data = Dataset.from_list(train_data)
test_data = Dataset.from_list(test_data)

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(bert_model)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


tokenized_train_data = train_data.map(preprocess_function, batched=True)
tokenized_test_data = test_data.map(preprocess_function, batched=True)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer


model = AutoModelForSequenceClassification.from_pretrained(
    bert_model, num_labels=3
).to(device)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=1,
    num_train_epochs=5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_data,
    # eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

result1 = trainer.predict(tokenized_test_data)
print(result1)
print('Len test data:', len(test_data))
# print(f"Execution time: {result1.test_runtime/len(test_data)}")
result1 = result1.predictions.argmax(axis=1)

y_test = [x["label"] for x in tokenized_test_data]

from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score

acc = accuracy_score(y_test, result1)
print(f'Accuracy: micro: {acc:.3f}')

p_micro = precision_score(y_test, result1, average='micro')
p_macro = precision_score(y_test, result1, average='macro')
print(f'Precision: micro-macro: {p_micro:.3f}-{p_macro:.3f}')

r_micro = recall_score(y_test, result1, average='micro')
r_macro = recall_score(y_test, result1, average='macro')
print(f'Recall: micro-macro: {r_micro:.3f}-{r_macro:.3f}')

f1_micro = f1_score(y_test, result1, average='micro')
f1_macro = f1_score(y_test, result1, average='macro')
print(f'F1: micro-macro: {f1_micro:.3f}-{f1_macro:.3f}')

print(classification_report(y_test, result1, target_names=["negative", "neutral", "positive"]))


n_true_pos = 0
n_true_neu = 0
n_true_neg = 0
for p, l in zip(result1, y_test):
    if p == l and l == 2:
        n_true_pos += 1
    elif p == l and l == 1:
        n_true_neu += 1
    elif p == l and l == 0:
        n_true_neg += 1
print(n_true_pos, n_true_neu, n_true_neg)
