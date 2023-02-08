from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
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
from sklearn.feature_extraction.text import TfidfVectorizer


class LinearSVMTFIDF:
    def __init__(self, model_path, load_inference_model=True):
        self.tweet_tokenizer = TweetTokenizer()
        self.stopwords = set(stopwords.words("english"))

        if load_inference_model:
            obj = pkl.load(open(model_path, "rb"))
            self.model = obj["model"]
            self.vectorizer = obj["vectorizer"]
        else:
            self.vectorizer = TfidfVectorizer(
                min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True
            )
            self.model = LinearSVC()

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
        sentence_vector = self.vectorizer.transform([processed_sentence])
        pred = self.model.predict(sentence_vector)[0]
        return self.id2label[pred]

    def train(self, train_data, output_model_name=None):
        texts = [x["text"] for x in train_data]
        labels = [x["label"] for x in train_data]

        train_vector = self.vectorizer.fit_transform(texts)
        self.model.fit(train_vector, labels)

        if output_model_name is None:
            output_model_name = f"linear_svm_tfidf-{time.time()}"

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
