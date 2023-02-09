from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
import time
import pickle as pkl
from sklearn.svm import LinearSVC
import dill


class LinearSVM:
    def __init__(self, model_path, load_inference_model=True):
        if load_inference_model:
            obj = pkl.load(open(model_path, "rb"))
            self.model = obj["model"]

            if "vectorizer" in obj:
                self.vectorizer = obj["vectorizer"]
        else:
            self.model = LinearSVC()

        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}

    def predict(self, sentence_vector):
        pred = self.model.predict([sentence_vector])[0]
        return self.id2label[pred]

    def train(self, train_vector, labels):
        self.model.fit(train_vector, labels)

    def save_model(self, vectorizer=None, output_model_name=None):
        if output_model_name is None:
            output_model_name = f"linear_svm_tfidf-{time.time()}"

        output_model_path = f"models/{output_model_name}"

        if dill.pickles(vectorizer):
            save_data = {"model": self.model, "vectorizer": vectorizer}
        else:
            save_data = {"model": self.model}
        pkl.dump(
            save_data,
            open(output_model_path, "wb"),
        )

        print(f"Model has been saved to {output_model_path}")

    def evaluate(self, test_vector, labels):
        result = self.model.predict(test_vector)

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
