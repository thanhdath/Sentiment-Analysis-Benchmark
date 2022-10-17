from .linear_svm_tfidf import LinearSVMTFIDF


def init_model(model_name):
    if model_name == "linear_svm_tfidf":
        model = LinearSVMTFIDF(f"models/{model_name}.pkl")

    return model
