import numpy as np
import pandas as pd
import json, nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

train_df = pd.read_csv("../data/combine/kfolds_0/train.csv", encoding="ISO-8859-1")
val_df = pd.read_csv("../data/combine/kfolds_0/test.csv", encoding="ISO-8859-1")
all_df = []
for values in train_df.values:
    values = values.tolist()
    values.append("train")
    all_df.append(values)
for values in val_df.values:
    values = values.tolist()
    values.append("test")
    all_df.append(values)
print(all_df[0])
columns = train_df.columns
print(columns.tolist())
total_data = pd.DataFrame(all_df, columns=train_df.columns.tolist() + ["split"])
total_data


# %%
with open("assets/contractions.json", "r") as f:
    contractions_dict = json.load(f)
contractions = contractions_dict["contractions"]

# %%
pd.set_option("display.max_colwidth", -1)

# %% [markdown]
# ##### Printing the dataset

# %%
total_data.head()

# %% [markdown]
# ##### Taking column names into variables

# %%
tweet = total_data.columns.values[2]
sentiment = total_data.columns.values[1]
tweet, sentiment

# %%
total_data.info()


# %%
def emoji(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :') , :O
    tweet = re.sub(r"(:\s?\)|:-\)|\(\s?:|\(-:|:\'\)|:O)", " positiveemoji ", tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r"(:\s?D|:-D|x-?D|X-?D)", " positiveemoji ", tweet)
    # Love -- <3, :*
    tweet = re.sub(r"(<3|:\*)", " positiveemoji ", tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-; , @-)
    tweet = re.sub(r"(;-?\)|;-?D|\(-?;|@-\))", " positiveemoji ", tweet)
    # Sad -- :-(, : (, :(, ):, )-:, :-/ , :-|
    tweet = re.sub(r"(:\s?\(|:-\(|\)\s?:|\)-:|:-/|:-\|)", " negetiveemoji ", tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', " negetiveemoji ", tweet)
    return tweet


# %% [markdown]
# ###### Define a function which will preprocess the tweets

# %%
import re


def process_tweet(tweet):
    tweet = tweet.lower()  # Lowercases the string
    tweet = re.sub("@[^\s]+", "", tweet)  # Removes usernames
    tweet = re.sub("((www\.[^\s]+)|(https?://[^\s]+))", " ", tweet)  # Remove URLs
    tweet = re.sub(r"\d+", " ", str(tweet))  # Removes all digits
    tweet = re.sub("&quot;", " ", tweet)  # Remove (&quot;)
    tweet = emoji(tweet)  # Replaces Emojis
    tweet = re.sub(r"\b[a-zA-Z]\b", "", str(tweet))  # Removes all single characters
    for word in tweet.split():
        if word.lower() in contractions:
            tweet = tweet.replace(
                word, contractions[word.lower()]
            )  # Replaces contractions
    tweet = re.sub(r"[^\w\s]", " ", str(tweet))  # Removes all punctuations
    tweet = re.sub(
        r"(.)\1+", r"\1\1", tweet
    )  # Convert more than 2 letter repetitions to 2 letter
    tweet = re.sub(r"\s+", " ", str(tweet))  # Replaces double spaces with single space
    return tweet


# %% [markdown]
# ###### Now make a new column for side by side comparison of new tweets vs old tweets

# %% [markdown]
# `Check this `**[Stackoverflow answer](https://stackoverflow.com/a/52674448/8141330)**` to know how to loop faster in python`

# %%
total_data["processed_tweet"] = np.vectorize(process_tweet)(total_data[tweet])

# %% [markdown]
# ###### Let's compare unprocessed tweets with the processed one

# %%
total_data.head(10)

# %% [markdown]
# ### Spelling correction

# %%
# from textblob import TextBlob
# total_data['processed_tweet'].apply(lambda x: str(TextBlob(x).correct()))
# total_data.head(10)

# %% [markdown]
# ### Tokenization

# %% [markdown]
# `We will be using string.split() instead of nltk.tokenize, check this `**[Stackoverflow answer](https://stackoverflow.com/a/35348340/8141330)**` for more information`

# %%
# tokenized_tweet = total_data['processed_tweet'].apply(lambda x: x.split())
# tokenized_tweet.head()

# %% [markdown]
# ### Stemming

# %% [markdown]
# **The below code is commented out because *Lemmatization* gives slightly better accuracy in this sentiment analysis than *Stemming*. If you want to check, then uncomment the code below, comment the Lemmatization code below and run the whole code again.**

# %%
# from nltk.stem.porter import *
# stemmer = PorterStemmer()

# tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])
# tokenized_tweet.head()

# %% [markdown]
# ### Lemmatization

# %% [markdown]
# `check this article on` **[Why use lemmatization over stemming](https://stackoverflow.com/questions/771918/how-do-i-do-word-stemming-or-lemmatization)**

# %%
# from nltk.stem.wordnet import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()

# tokenized_tweet = tokenized_tweet.apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
# tokenized_tweet.head()

# %% [markdown]
# ### Stop words

# %% [markdown]
# `Here is a list of NLTK stop words taken from` **[this GitHub link](https://gist.github.com/sebleier/554280)**
# <br/>
# *["i", "me", "my", "myself", "we", "our", "ours", "ourselves",
#  "you", "your", "yours", "yourself", "yourselves", "he", "him",
#  "his", "himself", "she", "her", "hers", "herself", "it", "its",
#  "itself", "they", "them", "their", "theirs", "themselves", "what",
#  "which", "who", "whom", "this", "that", "these", "those", "am", "is",
#  "are", "was", "were", "be", "been", "being", "have", "has", "had",
#  "having", "do", "does", "did", "doing", "a", "an", "the", "and",
#  "but", "if", "or", "because", "as", "until", "while", "of", "at",
#  "by", "for", "with", "about", "against", "between", "into", "through",
#  "during", "before", "after", "above", "below", "to", "from", "up",
#  "down", "in", "out", "on", "off", "over", "under", "again", "further",
#  "then", "once", "here", "there", "when", "where", "why", "how", "all",
#  "any", "both", "each", "few", "more", "most", "other", "some", "such",
#  "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
#  "s", "t", "can", "will", "just", "don", "should", "now"]*
# <br/>
# *We can't use every word from here. Because some words like `"no"`, `"nor"` etc. playes significant roles in sentiment.*
#
# ##### So we will be making our custom list of stopwords.

# %%
# stop_words = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves",
#             "you", "your", "yours", "yourself", "yourselves", "he", "him",
#             "his", "himself", "she", "her", "hers", "herself", "it", "its",
#             "itself", "they", "them", "their", "theirs", "themselves", "what",
#             "which", "who", "whom", "this", "that", "these", "those", "am", "is",
#             "are", "was", "were", "be", "been", "being", "have", "has", "had",
#             "having", "do", "does", "did", "doing", "a", "an", "the", "and",
#             "but", "if", "or", "because", "as", "until", "while", "of", "at",
#             "by", "for", "with", "about", "against", "between", "into", "through",
#             "during", "before", "after", "above", "below", "to", "from", "up",
#             "down", "in", "out", "on", "off", "over", "under", "again", "further",
#             "then", "once", "here", "there", "when", "where", "why", "how", "all",
#             "any", "both", "each", "few", "more", "most", "other", "some", "such",
#             "only", "own", "same", "so", "than", "too", "very",
#             "can", "will", "just", "should", "now"}

# %% [markdown]
# ##### Below is the in built stop words from nltk. But we can't use them. If you still want to see the words, You can uncomment the lines below
#

# %%
# nltk.download("stopwords")
# from nltk.corpus import stopwords
# stop_words = set(stopwords.words('english'))
# stop_words, nltk.corpus.stopwords.words('english')

# %% [markdown]
# ### Stiching

# %%
# for i in range(len(tokenized_tweet)):

#     # Below code is used for no stop word removal
#     tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

#     # Used for stop word removal
#     # (Below is commented out as sentiment analysis is giving better accuracy without removing stop words.
#     # If you still want to check, comment out the above line, uncomment the line below and run the code again.)

#     # tokenized_tweet[i] = ' '.join([word for word in tokenized_tweet[i] if word not in stop_words])


# total_data['processed_tweet'] = tokenized_tweet
# total_data.head()

# %% [markdown]
# # 2) Most used words

# %%
sentiments = ["Positive", "Neutral", "Negative"]
slices = [
    (total_data[sentiment] == "positive").sum(),
    (total_data[sentiment] == "neutral").sum(),
    (total_data[sentiment] == "negative").sum(),
]
colors = ["g", "r", "b"]
plt.pie(
    slices,
    labels=sentiments,
    colors=colors,
    startangle=90,
    shadow=True,
    explode=(0, 0.1, 0.1),
    radius=1.5,
    autopct="%1.2f%%",
)
plt.legend()
plt.show()

# %% [markdown]
# ## 2.2) Most used positive words

# %%
positive_words = " ".join(
    [
        text
        for text in total_data["processed_tweet"][total_data[sentiment] == "positive"]
    ]
)
wordcloud = WordCloud(
    width=800,
    height=500,
    random_state=21,
    max_font_size=110,
    background_color="rgba(255, 255, 255, 0)",
    mode="RGBA",
).generate(positive_words)
plt.figure(dpi=600)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Used Positive Words")
plt.savefig("assets/positive_words.png")
plt.show()

# %%
pos_coeff = sorted(wordcloud.words_.items(), key=lambda x: x[1], reverse=True)[:40]
for word, coeff in pos_coeff:
    print(word)
for word, coeff in pos_coeff:
    print(coeff)

# %% [markdown]
# ## 2.3) Most used negetive words

# %%
negetive_words = " ".join(
    [
        text
        for text in total_data["processed_tweet"][total_data[sentiment] == "negative"]
    ]
)
wordcloud = WordCloud(
    width=800,
    height=500,
    random_state=21,
    max_font_size=110,
    background_color="rgba(255, 255, 255, 0)",
    mode="RGBA",
).generate(negetive_words)
plt.figure(dpi=600)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Used Negative Words")
plt.savefig("assets/negative_words.png")
plt.show()

# %%
neg_coeff = sorted(wordcloud.words_.items(), key=lambda x: x[1], reverse=True)[:40]
for word, coeff in neg_coeff:
    print(word)
for word, coeff in neg_coeff:
    print(coeff)

# %%
negetive_words = " ".join(
    [text for text in total_data["processed_tweet"][total_data[sentiment] == "neutral"]]
)
wordcloud = WordCloud(
    width=800,
    height=500,
    random_state=21,
    max_font_size=110,
    background_color="rgba(255, 255, 255, 0)",
    mode="RGBA",
).generate(negetive_words)
plt.figure(dpi=600)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Used Neutral Words")
plt.savefig("assets/neutral_words.png")
plt.show()

# %% [markdown]
# #### See the word `lol`. It is used both in positive and negetive(sarcastic) sentiments. We still can't classify sarcasm.

# %% [markdown]
# # 3) Feature extraction (vectorization)

# %% [markdown]
# ## N-grams included (Unigram, Bigram, Trigram)

# %% [markdown]
# `check this article on` **[How to Prepare Text Data for Machine Learning with scikit-learn](https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/)**

# %% [markdown]
# `check this article on` **[countvectorizer and tf-idf](https://www.kaggle.com/divsinha/sentiment-analysis-countvectorizer-tf-idf)**

# %% [markdown]
# *Tf-idf* is different from *CountVectorizer*. *CountVectorizer* gives equal weightage to all the words, i.e. a word is converted to a column (in a dataframe for example) and for each document, it is equal to 1 if it is present in that doc else 0.
# Apart from giving this information, *Tf-idf* says how important that word is to that document with respect to the corpus.

# %% [markdown]
# ## 3.1) Count vectorizer

# %% [markdown]
# As we all know, all machine learning algorithms are good with numbers; we have to extract or convert the text data into numbers without losing much of the information. One way to do such transformation is *Bag-Of-Words (BOW)* which gives a number to each word but that is very inefficient. So, a way to do it is by *CountVectorizer*: it counts the number of words in the document i.e it converts a collection of text documents to a matrix of the counts of occurences of each word in the document.

# %%
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(ngram_range=(1, 2))  # Unigram and Bigram
final_vectorized_data = count_vectorizer.fit_transform(total_data["processed_tweet"])
final_vectorized_data

# %% [markdown]
# ## 3.2) Tf-Idf vectorizer

# %% [markdown]
# *TF-IDF (stands for Term-Frequency-Inverse-Document Frequency)* weights down the common words occuring in almost all the documents and give more importance to the words that appear in a subset of documents. *TF-IDF* works by penalising these common words by assigning them lower weights while giving importance to some rare words in a particular document. ***Rare terms are more informative than frequent terms.***
#
# ![](assets/tfidf.png)
#
# <br/>

# %% [markdown]
# **The below code is commented out because *CountVectorizer* gives better accuracy in this sentiment analysis than *tf-idf*. If you want to check, then uncomment the code below, comment the countVectorizer code above and run the whole code again.**
# <br/>
#

# %%
# from sklearn.feature_extraction.text import TfidfVectorizer

# tf_idf_vectorizer = TfidfVectorizer(use_idf=True,ngram_range=(1,3))
# final_vectorized_data = tf_idf_vectorizer.fit_transform(total_data['processed_tweet'])

# final_vectorized_data

# %% [markdown]
# # 4) Splitting

# %% [markdown]
# ##### Splitting train data to test accuracy

# %%
final_vectorized_data

train_inds = [
    i for i, split in enumerate(total_data["split"].values) if split == "train"
]
test_inds = [i for i, split in enumerate(total_data["split"].values) if split == "test"]

X_train = final_vectorized_data[train_inds]
X_test = final_vectorized_data[test_inds]
y_train = total_data[sentiment].values[train_inds]
y_test = total_data[sentiment].values[test_inds]
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(final_vectorized_data, total_data[sentiment],
#                                                     test_size=0.2, random_state=69)

# %% [markdown]
# ##### Printing splitted dataset sizes

# %%
print("X_train_shape : ", X_train.shape)
print("X_test_shape : ", X_test.shape)
print("y_train_shape : ", y_train.shape)
print("y_test_shape : ", y_test.shape)

# %% [markdown]
# # 5) Train and predict

# %% [markdown]
# ## 5.1) Naive_bayes

# %% [markdown]
# ### Bayes theorem :
# ![](assets/bayes_formula.jpg)

# %% [markdown]
# ### There are some popular classifiers under Naive Bayes
# * **Bernoulli Naive Bayes**
# * **Gaussian Naive Bayes classifier**
# * **Multinomial Naive Bayes**

# %% [markdown]
# #### We will use Multinomial Naive Bayes classifier

# %%
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes Classifier

model_naive = MultinomialNB().fit(X_train, y_train)
predicted_naive = model_naive.predict(X_test)

# %%
import pickle as pkl
import os

os.makedirs("../models/", exist_ok=True)

obj = {
    "model": model_naive,
    "contractions": contractions,
    "vectorizer": count_vectorizer,
}
pkl.dump(obj, open("../models/naive_bayes.pkl", "wb"))

# %% [markdown]
# ##### Print Confusion matrix

# %%
from sklearn.metrics import confusion_matrix

plt.figure(dpi=80)
mat = confusion_matrix(y_test, predicted_naive)
sns.heatmap(mat.T, annot=True, fmt="d", cbar=False)

plt.title("Confusion Matrix for Naive Bayes")
plt.xlabel("true label")
plt.ylabel("predicted label")
plt.savefig("assets/confusion_matrix.png")
plt.show()

# %% [markdown]
# ## Find out accuracy of our prediction

# %%
from sklearn.metrics import accuracy_score

score_naive = accuracy_score(predicted_naive, y_test)
print("Accuracy with Naive-bayes: ", score_naive)

# %%
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)

acc = accuracy_score(y_test, predicted_naive)
print(f"Accuracy: micro: {acc:.3f}")

p_micro = precision_score(y_test, predicted_naive, average="micro")
p_macro = precision_score(y_test, predicted_naive, average="macro")
print(f"Precision: micro-macro: {p_micro:.3f}-{p_macro:.3f}")

r_micro = recall_score(y_test, predicted_naive, average="micro")
r_macro = recall_score(y_test, predicted_naive, average="macro")
print(f"Recall: micro-macro: {r_micro:.3f}-{r_macro:.3f}")

f1_micro = f1_score(y_test, predicted_naive, average="micro")
f1_macro = f1_score(y_test, predicted_naive, average="macro")
print(f"F1: micro-macro: {f1_micro:.3f}-{f1_macro:.3f}")


# %% [markdown]
# # 6) Precision, Recall, and Accuracy
#
# ##### Precision, recall, and accuracy are standard metrics used to evaluate the performance of a classifier.
#
# * Precision measures how many texts were predicted correctly as belonging to a given category out of all of the texts that were predicted (correctly and incorrectly) as belonging to the category.
#
# * Recall measures how many texts were predicted correctly as belonging to a given category out of all the texts that should have been predicted as belonging to the category. We also know that the more data we feed our classifiers with, the better recall will be.
#
# * Accuracy measures how many texts were predicted correctly (both as belonging to a category and not belonging to the category) out of all of the texts in the corpus.
#
# ##### Most frequently, precision and recall are used to measure performance since accuracy alone does not say much about how good or bad a classifier is.

# %%
from sklearn.metrics import classification_report

print(classification_report(y_test, predicted_naive))

result1 = predicted_naive
n_true_pos = 0
n_true_neu = 0
n_true_neg = 0
for p, l in zip(result1, y_test):
    if p == l and l == "positive":
        n_true_pos += 1
    elif p == l and l == "neutral":
        n_true_neu += 1
    elif p == l and l == "negative":
        n_true_neg += 1
print(n_true_pos, n_true_neu, n_true_neg)


# %% [markdown]
# `Check this link to learn` **[ROC, precision & recall curves](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/)**
