from .word2vec_vectorizer import Word2VecVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def init_vectorizer(name):
    if name == "tfidf":
        vectorizer = TfidfVectorizer(
            min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True
        )
    elif name == "word2vec":
        vectorizer = Word2VecVectorizer()
    elif name == "bow":
        vectorizer = CountVectorizer(ngram_range=(1, 2))  # Unigram and Bigram
    elif name == "pass":
        vectorizer = Vectorizer() # just an abstract class that doesn't do anything
    else:
        raise NotImplementedError()

    return vectorizer

class Vectorizer():
    def fit(self, train_data):
        pass

    def transform(self, data):
        return data
