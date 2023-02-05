from .linear_svm_tfidf import LinearSVMTFIDF
from .random_forest import RandomForest

def init_model(model_name):
    if model_name == "linear_svm_tfidf":
        model = LinearSVMTFIDF(f"models/{model_name}.pkl")
    elif model_name == "random_forest":
        model = RandomForest(f"models/{model_name}.pkl")
        
    return model
