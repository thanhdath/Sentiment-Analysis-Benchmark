# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 03:55:04 2017

@author: YJ
"""

from importlib.resources import read_text
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import string
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
import time

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

tweet_train_path = "data/combine/kfolds_0/train.csv"
tweet_test_path = "data/combine/kfolds_0/test.csv"

tweet_tokenizer = TweetTokenizer()


def read_text_and_label(file):
    df = pd.read_csv(file)
    parsed_tweet = []
    tweet_target = []

    for _, label, text in df.values:
        l = tweet_tokenizer.tokenize(text)
        l = [x.lower() for x in l]
        filtered_sentence = [
            w
            for w in l
            if not w in stop
            and not w in string.punctuation
            and (w[0] != "@" and w[0] != "#" and w[:4] != "http")
        ]
        parsed_tweet.append(" ".join(filtered_sentence))

        label_number = 0
        if label == "negative":
            label_number = 0
        elif label == "neutral":
            label_number = 1
        elif label == "positive":
            label_number = 2
        tweet_target.append(label_number)
    tweet_target = np.array(tweet_target)
    return parsed_tweet, tweet_target


# stop words

stop = set(stopwords.words("english"))


# label the data

total_svm = 0

""" 
80% Training , 20% Testing
"""
stime = time.time()
X_train, y_train = read_text_and_label(tweet_train_path)
X_test, y_test = read_text_and_label(tweet_test_path)

vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)

# Returns a feature vectors matrix having a fixed length tf-idf weighted word count feature
# for each document in training set. aka Term-document matrix

train_corpus_tf_idf = vectorizer.fit_transform(X_train)
test_corpus_tf_idf = vectorizer.transform(X_test)

model1 = LinearSVC()
model1.fit(train_corpus_tf_idf, y_train)
etime = time.time()

print(f"training time: {etime-stime:.3f}")


result1 = model1.predict(test_corpus_tf_idf)
total_svm = total_svm + sum(y_test == result1)

total_svm = 0

acc = accuracy_score(y_test, result1)
print(f'Accuracy: micro: {acc:.3f}')

p_micro = precision_score(y_test, result1, average='micro')
p_macro = precision_score(y_test, result1, average='macro')
print(f'Precision: micro-macro: {p_micro:.3f}-{p_macro:.3f}')

r_micro = recall_score(y_test, result1, average='micro')
r_macro = recall_score(y_test, result1, average='macro')
print(f'Recall: micro-macro: {r_micro:.3f}-{r_macro:.3f}')

f1_micro = f1_score(y_test, result1, average='micro')
f1_macro = f1_score(y_test, result1, average='macro')
print(f'F1: micro-macro: {f1_micro:.3f}-{f1_macro:.3f}')

print(
    classification_report(
        y_test, result1, target_names=["negative", "neutral", "positive"]
    )
)

n_true_pos = 0
n_true_neu = 0
n_true_neg = 0
for p, l in zip(result1, y_test):
    if p == l and l == 2:
        n_true_pos += 1
    elif p == l and l == 1:
        n_true_neu += 1
    elif p == l and l == 0:
        n_true_neg += 1
print(n_true_pos, n_true_neu, n_true_neg)


print("Test execution time")

stime = time.time()
for i in range(100):
    test = vectorizer.transform([X_test[i]])
    model1.predict(test)
etime = time.time()
ave_time = (etime - stime) / 100
print(f"Execution time: {ave_time}")
