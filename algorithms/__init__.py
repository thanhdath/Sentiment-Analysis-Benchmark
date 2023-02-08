from .linear_svm_tfidf import LinearSVMTFIDF
from .linear_svm_word2vec import LinearSVMWord2Vec
from .logistic_regressor_word2vec import LogisticRegressorWord2Vec
from .bert import LMForSequenceClassification
from .naive_bayes import NaiveBayes
from .random_forest import RandomForest


def init_model(model_name):
    if model_name == "linear_svm_tfidf":
        model = LinearSVMTFIDF(f"models/{model_name}.pkl")
    elif model_name == "linear_svm_word2vec":
        model = LinearSVMWord2Vec(f"models/{model_name}.pkl")
    elif model_name == "logistic_regressor_word2vec":
        model = LogisticRegressorWord2Vec(f"models/{model_name}.pkl")
    elif model_name in ["distilbert-base-uncased", "bert-base-uncased", "roberta-base"]:
        model = LMForSequenceClassification(model_name)
    elif model_name == "naive_bayes":
        model = NaiveBayes(f"models/{model_name}.pkl")
    elif model_name == "random_forest":
        model = RandomForest(f"models/{model_name}.pkl")
    else:
        raise NotImplementedError()

    return model


def init_trainer(model_name):
    if model_name == "linear_svm_tfidf":
        model = LinearSVMTFIDF(f"models/{model_name}.pkl")
    elif model_name == "linear_svm_word2vec":
        model = LinearSVMWord2Vec(f"models/{model_name}.pkl")
    elif model_name == "logistic_regressor_word2vec":
        model = LogisticRegressorWord2Vec(f"models/{model_name}.pkl")
    elif model_name in ["distilbert-base-uncased", "bert-base-uncased", "roberta-base"]:
        model = LMForSequenceClassification(model_name, load_inference_model=False)
    elif model_name == "naive_bayes":
        model = NaiveBayes(f"models/{model_name}.pkl")
    elif model_name == "random_forest":
        model = RandomForest(f"models/{model_name}.pkl")
    else:
        raise NotImplementedError()

    return model
