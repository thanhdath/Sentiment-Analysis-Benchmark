import pickle as pkl
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)


class RandomForest:
    def __init__(self, model_path, load_inference_model=True):
        self.stopwords = nltk.corpus.stopwords.words("english")

        if load_inference_model:
            obj = pkl.load(open(model_path, "rb"))
            self.model = obj["model"]
            self.vectorizer = obj["vectorizer"]
        else:
            self.vectorizer = TfidfVectorizer()
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=None, n_jobs=-1
            )

        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}

    def remove_noisy_words(self, sentence):
        words = self.tweet_tokenizer.tokenize(sentence)
        words = [
            w for w in words if w not in self.stopwords and w not in string.punctuation
        ]
        sentence = " ".join(words)
        return sentence

    def predict(self, sentence):
        processed_sentence = self.remove_noisy_words(sentence)
        # text_len = len(processed_sentence.split())

        tfidf_vector = self.vectorizer.transform([processed_sentence]).toarray()
        # sentence_vector = np.concatenate([np.array([[text_len]]), tfidf_vector], axis=1)

        pred = self.model.predict(tfidf_vector)[0]
        return pred

    def train(self, train_data, output_model_name=None):
        texts = [x["text"] for x in train_data]
        labels = [x["label"] for x in train_data]

        train_vector = self.vectorizer.fit_transform(texts)
        self.model.fit(train_vector, labels)

        if output_model_name is None:
            output_model_name = f"random-forest-{time.time()}"

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
