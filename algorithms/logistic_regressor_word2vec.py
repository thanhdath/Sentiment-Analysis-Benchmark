from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import string
import pickle as pkl
import fasttext
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
import time
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)


class LogisticRegressorWord2Vec:
    def __init__(
        self,
        model_path,
        fasttext_wv_path="models/cc.en.300.bin",
        fasttext_wv_dim=300,
        load_inference_model=True,
    ):
        self.tweet_tokenizer = TweetTokenizer()
        self.stopwords = set(stopwords.words("english"))
        self.fasttext = fasttext.load_model(fasttext_wv_path)
        self.fasttext_wv_dim = fasttext_wv_dim

        if load_inference_model:
            obj = pkl.load(open(model_path, "rb"))
            self.model = obj["model"]
            self.vectorizer = obj["vectorizer"]
        else:
            self.vectorizer = TfidfVectorizer(
                min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True
            )
            self.model = linear_model.LogisticRegression(C=1e5)

        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}

    def init_word2vec_features(self, text):
        tfidf_vec = self.vectorizer.transform([text]).toarray()[0]
        feature_index = tfidf_vec.nonzero()[0]
        feature_names = self.vectorizer.get_feature_names()
        score_dict = {feature_names[i]: tfidf_vec[i] for i in feature_index}

        average_vec = np.zeros(self.fasttext_wv_dim)
        for word in text.split():
            if word in score_dict:
                weight = score_dict[word]
            else:
                weight = 1.0
            average_vec += (self.fasttext.get_word_vector(word) * weight) / len(
                text.split()
            )
        return average_vec

    def remove_noisy_words(self, sentence):
        words = self.tweet_tokenizer.tokenize(sentence)
        words = [
            w for w in words if w not in self.stopwords and w not in string.punctuation
        ]
        sentence = " ".join(words)
        return sentence

    def predict(self, sentence):
        processed_sentence = self.remove_noisy_words(sentence)
        sentence_vector = self.init_word2vec_features(processed_sentence)
        pred = self.model.predict([sentence_vector])[0]
        return self.id2label[pred]

    def train(self, train_data, output_model_name=None):
        texts = [x["text"] for x in train_data]
        labels = [x["label"] for x in train_data]

        train_vector = self.vectorizer.fit_transform(texts)
        train_vector = self.init_word2vec_features(train_vector)
        self.model.fit(train_vector, labels)

        if output_model_name is None:
            output_model_name = f"logistic_regressor_word2vec-{time.time()}"

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
        test_vector = self.init_word2vec_features(test_vector)
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
