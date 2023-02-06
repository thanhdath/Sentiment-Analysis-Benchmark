from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import string
import pickle as pkl
import fasttext
import numpy as np

class LogisticRegressorWord2Vec:
    def __init__(self, model_path, fasttext_wv_path='models/cc.en.300.bin'):
        self.tweet_tokenizer = TweetTokenizer()
        self.stopwords = set(stopwords.words("english"))
        
        self.fasttext = fasttext.load_model(fasttext_wv_path)

        obj = pkl.load(open(model_path, "rb"))
        self.model = obj["model"]
        self.vectorizer = obj["vectorizer"]

        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}
        
    def init_word2vec_features(self, text):
        tfidf_vec = self.vectorizer.transform([text]).toarray()[0]
        feature_index = tfidf_vec.nonzero()[0]
        feature_names = self.vectorizer.get_feature_names()
        score_dict = {
            feature_names[i]: tfidf_vec[i] for i in feature_index
        }
        
        average_vec = np.zeros(300)
        for word in text.split():
            if word in score_dict:
                weight = score_dict[word]
            else:
                weight = 1.0
            average_vec += (self.fasttext.get_word_vector(word) * weight) / len(text.split())
        return average_vec


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
        sentence_vector = self.init_word2vec_features(processed_sentence)
        pred = self.model.predict([sentence_vector])[0]
        return self.id2label[pred]

