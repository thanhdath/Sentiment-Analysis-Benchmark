import pickle as pkl
import re
from sklearn.feature_extraction.text import CountVectorizer


class NaiveBayes:
    def __init__(self, model_path):
        obj = pkl.load(open(model_path, "rb"))

        self.contractions = obj["contractions"]
        self.model = obj["model"]
        self.vectorizer = obj["vectorizer"]

    def predict(self, sentence):
        processed_sentence = self.process_tweet(sentence)
        sentence_vector = self.vectorizer.transform([processed_sentence])
        pred = self.model.predict(sentence_vector)[0]
        return pred

    def emoji(self, tweet):
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

    def process_tweet(self, tweet):
        tweet = tweet.lower()  # Lowercases the string
        tweet = re.sub("@[^\s]+", "", tweet)  # Removes usernames
        tweet = re.sub("((www\.[^\s]+)|(https?://[^\s]+))", " ", tweet)  # Remove URLs
        tweet = re.sub(r"\d+", " ", str(tweet))  # Removes all digits
        tweet = re.sub("&quot;", " ", tweet)  # Remove (&quot;)
        tweet = self.emoji(tweet)  # Replaces Emojis
        tweet = re.sub(r"\b[a-zA-Z]\b", "", str(tweet))  # Removes all single characters
        for word in tweet.split():
            if word.lower() in self.contractions:
                tweet = tweet.replace(
                    word, self.contractions[word.lower()]
                )  # Replaces contractions
        tweet = re.sub(r"[^\w\s]", " ", str(tweet))  # Removes all punctuations
        tweet = re.sub(
            r"(.)\1+", r"\1\1", tweet
        )  # Convert more than 2 letter repetitions to 2 letter
        tweet = re.sub(
            r"\s+", " ", str(tweet)
        )  # Replaces double spaces with single space
        return tweet
