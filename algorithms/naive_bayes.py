import pickle as pkl
import re
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import json, nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import time
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)


class NaiveBayes:
    def __init__(self, model_path, load_inference_model=True):
        if load_inference_model:
            obj = pkl.load(open(model_path, "rb"))

            self.contractions = obj["contractions"]
            self.model = obj["model"]
            self.vectorizer = obj["vectorizer"]
        else:
            self.contractions = json.load(open("data/assets/contractions.json"))
            self.vectorizer = CountVectorizer(ngram_range=(1, 2))  # Unigram and Bigram
            self.model = MultinomialNB()

    def predict(self, processed_sentence):
        sentence_vector = self.vectorizer.transform([processed_sentence])
        pred = self.model.predict(sentence_vector)[0]
        return pred

    def train(self, train_data, output_model_name=None):
        texts = [x["text"] for x in train_data]
        labels = [x["label"] for x in train_data]

        train_vector = self.vectorizer.fit_transform(texts)
        self.model.fit(train_vector, labels)

        if output_model_name is None:
            output_model_name = f"naive-bayes-{time.time()}"

        output_model_path = f"models/{output_model_name}"
        pkl.dump(
            {"model": self.model, "vectorizer": self.vectorizer},
            open(output_model_path, "wb"),
        )

        print(f"Model has been saved to {output_model_path}")

    def evaluate(self, test_data):
        texts = [x["text"] for x in test_data]
        y_test = [x["label"] for x in test_data]

        test_vector = self.vectorizer.transform(texts)
        result = self.model.predict(test_vector)

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
                y_test, result, target_names=["negative", "neutral", "positive"]
            )
        )
