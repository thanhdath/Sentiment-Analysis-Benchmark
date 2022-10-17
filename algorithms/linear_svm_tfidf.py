from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import string
import pickle as pkl


class LinearSVMTFIDF:
    def __init__(self, model_path):
        self.tweet_tokenizer = TweetTokenizer()
        self.stopwords = set(stopwords.words("english"))

        obj = pkl.load(open("models/linear_svm_tfidf.pkl", "rb"))
        self.model = obj["model"]
        self.vectorizer = obj["vectorizer"]

        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}

    def process_sentence(self, sentence):
        l = self.tweet_tokenizer.tokenize(sentence)
        l = [x.lower() for x in l]

        filtered_sentence = " ".join(
            [
                w
                for w in l
                if not w in self.stopwords
                and not w in string.punctuation
                and (w[0] != "@" and w[0] != "#" and w[:4] != "http")
            ]
        )
        return filtered_sentence

    def predict(self, sentence):
        processed_sentence = self.process_sentence(sentence)
        sentence_vector = self.vectorizer.transform([processed_sentence])
        pred = self.model.predict(sentence_vector)[0]
        return self.id2label[pred]
