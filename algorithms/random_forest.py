import pickle as pkl
import nltk
import re
import string
import numpy as np

class RandomForest:
    def __init__(self, model_path):
        self.stopword = nltk.corpus.stopwords.words('english')
        self.wn = nltk.WordNetLemmatizer()
  
        obj = pkl.load(open(model_path, "rb"))
        self.model = obj["model"]
        self.vectorizer = obj["vectorizer"]
#         self.id2label = {0: "negative", 1: "neutral", 2: "positive"}

    def clean_tokenize_text(self, text):
        # Remove mentions starting with @
        text = re.sub('@\w+\s', '', text)

        # Remove punctuation
        text = ''.join([char for char in text if char not in string.punctuation])

        text = text.lower()

        # Tokenize the text
        tokens = re.split('\W+', text)

        # Remove stopwords
        text = [word for word in tokens if word not in self.stopword]

        # Lemmatize the words
        text = [self.wn.lemmatize(word) for word in text]
        text = ' '.join(text)

        # Return text
        return text

    def predict(self, sentence):
        processed_sentence = self.clean_tokenize_text(sentence)
        text_len = len(processed_sentence.split())
        
        tfidf_vector = self.vectorizer.transform([processed_sentence]).toarray()
        sentence_vector = np.concatenate([np.array([[text_len]]), tfidf_vector], axis=1)
       
        pred = self.model.predict(sentence_vector)[0]
        return pred
#         return self.id2label[pred]
