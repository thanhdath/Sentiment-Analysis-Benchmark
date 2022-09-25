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
from sklearn.metrics import classification_report
import pandas as pd

tweet_train_path = 'data/combine/kfolds_0/train.csv'
tweet_test_path = 'data/combine/kfolds_0/test.csv'
path = ''

tweet_tokenizer = TweetTokenizer()

def read_text_and_label(file):
    df = pd.read_csv(file)
    parsed_tweet = []
    tweet_target = []

    for _, label, text in df.values:
        l = tweet_tokenizer.tokenize(text)
        l = [x.lower() for x in l]
        filtered_sentence = [w for w in l if not w in stop and not w in string.punctuation 
                            and ( w[0] != '@' and w[0] != '#' and w[:4] != 'http' )]
        parsed_tweet.append(' '.join(filtered_sentence))

        label_number = 0
        if label == 'negative':
            label_number = 0
        elif label == 'neutral':
            label_number = 1
        elif label == 'positive':
            label_number = 2
        tweet_target.append(label_number)
    tweet_target = np.array(tweet_target)
    return parsed_tweet, tweet_target

# stop words

stop = set(stopwords.words('english'))


# label the data

total_svm = 0

""" 
80% Training , 20% Testing
"""

X_train, y_train = read_text_and_label(tweet_train_path)
X_test, y_test = read_text_and_label(tweet_test_path)


model = KeyedVectors.load_word2vec_format(path, binary=True)

""" 
80% Training , 20% Testing
"""


# Initialize word2vec_feature vector
total_svm = 0

def init_word2vec_features(text_list):
    word2vec_feature = []

    # adds the word2vec average
    for tweet in text_list:
        average_vec = np.zeros(300)
        for word in tweet:
            if word in model.wv:
                average_vec += (model.wv[word] / len(tweet))
            else:
                pass
        word2vec_feature.append(average_vec)
    return word2vec_feature

X_train = init_word2vec_features(X_train)
X_test = init_word2vec_features(X_test)

svc_model = LinearSVC()
svc_model.fit(X_train, y_train)

result1 = svc_model.predict(X_test)

print(classification_report(y_test, result1, target_names=['negative', 'neutral', 'positive']))
