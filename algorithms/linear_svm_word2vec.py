import gensim
import numpy as np

from gensim.models.keyedvectors import KeyedVectors

from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import string
import csv
import numpy as np
from sklearn.svm import LinearSVC
import pandas as pd
import time
import fasttext

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import string
import pickle as pkl


class LinearSVMWord2Vec:
    def __init__(self, model_path):
        self.tweet_tokenizer = TweetTokenizer()
        self.stopwords = set(stopwords.words("english"))
        self.fasttext_model = fasttext.load_model("models/cc.en.300.bin")

        obj = pkl.load(open("models/linear_svm_word2vec.pkl", "rb"))
        self.model = obj["model"]

        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}
        
    def get_word2vec_feature(self, tweet):
        word2vec_feature = []

        tweet = tweet.split()
        average_vec = np.zeros(300)
        for word in tweet:
            average_vec += self.fasttext_model.get_word_vector(word) / len(tweet)
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
        sentence_vector = self.get_word2vec_feature(processed_sentence)
        pred = self.model.predict([sentence_vector])[0]
        return self.id2label[pred]

