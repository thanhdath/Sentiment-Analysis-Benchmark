from .linear_svm_tfidf import LinearSVMTFIDF
from .linear_svm_word2vec import LinearSVMWord2Vec
from .bert import DocumentEmbedding
from .naive_bayes import NaiveBayes


def init_model(model_name):
    if model_name == "linear_svm_tfidf":
        model = LinearSVMTFIDF(f"models/{model_name}.pkl")
    elif model_name == "linear_svm_word2vec":
        model = LinearSVMWord2Vec(f"models/{model_name}.pkl")
    elif model_name in ["distilbert-base-uncased", "bert-base-uncased", "roberta-base"]:
        model = DocumentEmbedding(model_name)
    elif model_name == "naive_bayes":
        model = NaiveBayes(f"models/{model_name}.pkl")
    elif model_name == "random_forest":
        model = RandomForest(f"models/{model_name}.pkl")
    else:
        raise NotImplementedError()

    return model
