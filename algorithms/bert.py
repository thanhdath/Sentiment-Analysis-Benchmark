from transformers import AutoTokenizer
import torch
from transformers import AutoModelForSequenceClassification
import numpy as np
from transformers import AutoTokenizer
import torch
from datasets import Dataset
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import time
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
import os


class LMForSequenceClassification:
    def __init__(self, model_name, load_inference_model=True):
        if load_inference_model:
            # Load pretrained model for inference
            pretrained_model_path = f"models/{model_name}"
        else:
            # Initialize from original model, finetuning with function train()
            pretrained_model_path = model_name

        if "uncased" in model_name:
            self.lower = True
        else:
            self.lower = False

        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_path, num_labels=3
        )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.load_inference_model = load_inference_model

        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}

    def predict(self, sentence, output_dir):
        if self.lower:
            sentence = sentence.lower()

        tokenized_sentence = self.tokenizer(
            sentence, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(**tokenized_sentence).logits[0]
        label = logits.argmax(axis=-1)
        return self.id2label[int(label)]

    def preprocess_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True)

    def train(self, train_texts, train_labels, device=None, output_model_name=None):
        """
        train_data: list of [{'text', 'label_number'}]
        device: e.g. cuda:0, cpu, cuda:1
        """
        train_data = [
            {"text": text, "label": label}
            for text, label in zip(train_texts, train_labels)
        ]

        train_data = Dataset.from_list(train_data)

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        tokenized_train_data = train_data.map(self.preprocess_function, batched=True)

        model = self.model.to(device)

        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=1,
            num_train_epochs=5,
            weight_decay=0.01,
        )

        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_data,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

        self.trainer.train()

    def save_model(self, output_model_name, **kwargs):
        if output_model_name is None:
            output_model_name = f"{self.model_name}-{time.time()}"

        output_model_path = f"models/{output_model_name}"
        os.makedirs("models/", exist_ok=True)
        self.trainer.save_model(output_model_path)

        print(f"Model has been saved to {output_model_path}")

    def evaluate(self, test_texts, test_labels):
        test_data = [
            {"text": text, "label": label}
            for text, label in zip(test_texts, test_labels)
        ]
        test_data = Dataset.from_list(test_data)
        tokenized_test_data = test_data.map(self.preprocess_function, batched=True)

        training_args = TrainingArguments(
            output_dir="./results", per_device_eval_batch_size=2, num_train_epochs=0
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

        stime = time.time()
        result = trainer.predict(tokenized_test_data)
        inference_time = time.time() - stime

        result = result.predictions.argmax(axis=1)

        y_test = [x["label"] for x in tokenized_test_data]

        acc = accuracy_score(y_test, result)
        print(f"Accuracy: micro: {acc:.3f}")

        p_micro = precision_score(y_test, result, average="micro")
        p_macro = precision_score(y_test, result, average="macro")
        print(f"Precision: micro-macro: {p_micro:.3f}-{p_macro:.3f}")

        r_micro = recall_score(y_test, result, average="micro")
        r_macro = recall_score(y_test, result, average="macro")
        print(f"Recall: micro-macro: {r_micro:.3f}-{r_macro:.3f}")

        f1_micro = f1_score(y_test, result, average="micro")
        f1_macro = f1_score(y_test, result, average="macro")
        print(f"F1: micro-macro: {f1_micro:.3f}-{f1_macro:.3f}")

        print(
            classification_report(
                y_test,
                result,
                target_names=[self.id2label[x] for x in range(len(self.id2label))],
            )
        )

        return {
            "acc": acc,
            "p_micro": p_micro,
            "p_macro": p_macro,
            "r_micro": r_micro,
            "r_macro": r_macro,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "inference_time": inference_time,
        }
