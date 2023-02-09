import numpy as np
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import string
import fasttext


class Word2VecVectorizer:
    def __init__(self, fasttext_wv_path="models/cc.en.300.bin", fasttext_wv_dim=300):
        self.tweet_tokenizer = TweetTokenizer()
        self.stopwords = set(stopwords.words("english"))
        print(f"Loading fasttext model {fasttext_wv_path}")
        self.fasttext_model = fasttext.load_model(fasttext_wv_path)
        self.fasttext_wv_dim = fasttext_wv_dim

    def get_word2vec_feature(self, tweet):
        tweet = self.remove_noisy_words(tweet)
        tweet = tweet.split()
        average_vec = np.zeros(self.fasttext_wv_dim)
        for word in tweet:
            average_vec += self.fasttext_model.get_word_vector(word) / len(tweet)
        return average_vec

    def remove_noisy_words(self, sentence):
        words = self.tweet_tokenizer.tokenize(sentence)
        words = [
            w for w in words if w not in self.stopwords and w not in string.punctuation
        ]
        sentence = " ".join(words)
        return sentence

    def fit(self, train_texts):
        pass

    def transform(self, texts):
        vector = [self.get_word2vec_feature(text) for text in texts]
        vector = np.array(vector)
        return vector
