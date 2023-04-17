import pickle as pkl
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
import dill
import os


class RandomForest:
    def __init__(self, model_path, load_inference_model=True):
        if load_inference_model:
            obj = pkl.load(open(model_path, "rb"))
            self.model = obj["model"]

            if "vectorizer" in obj:
                self.vectorizer = obj["vectorizer"]
        else:
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=None, n_jobs=-1
            )

        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}

    def train(self, train_vector, labels):
        self.model.fit(train_vector, labels)

    def save_model(self, vectorizer=None, output_model_name=None):
        if output_model_name is None:
            output_model_name = f"linear_svm_tfidf-{time.time()}"

        output_model_path = f"models/{output_model_name}"
        os.makedirs("models/", exist_ok=True)

        if dill.pickles(vectorizer):
            save_data = {"model": self.model, "vectorizer": vectorizer}
        else:
            save_data = {"model": self.model}

        with open(output_model_path, "wb") as fp:
            pkl.dump(
                save_data,
                fp,
            )

        print(f"Model has been saved to {output_model_path}")

    def evaluate(self, test_vector, labels):
        stime = time.time()
        result = self.model.predict(test_vector)
        inference_time = time.time() - stime

        acc = accuracy_score(labels, result)
        print(f"Accuracy: micro: {acc:.3f}")

        p_micro = precision_score(labels, result, average="micro")
        p_macro = precision_score(labels, result, average="macro")
        print(f"Precision: micro-macro: {p_micro:.3f}-{p_macro:.3f}")

        r_micro = recall_score(labels, result, average="micro")
        r_macro = recall_score(labels, result, average="macro")
        print(f"Recall: micro-macro: {r_micro:.3f}-{r_macro:.3f}")

        f1_micro = f1_score(labels, result, average="micro")
        f1_macro = f1_score(labels, result, average="macro")
        print(f"F1: micro-macro: {f1_micro:.3f}-{f1_macro:.3f}")

        print(
            classification_report(
                labels, result, target_names=["negative", "neutral", "positive"]
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
