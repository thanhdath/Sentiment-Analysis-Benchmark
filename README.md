# Sentiment-Analysis-Benchmark

## Setup environment
```
conda create -p .env/ python=3.9
pip install -r requirements.txt
```

## Training and Evaluate

Check the notebook TrainingAndEvaluate.ipynb

**Data format**: train and test file in csv format, each file must contain 3 columns: tweet_id, label and text (see example in TrainingAndEvalute.ipynb).

**Supported algorithms**:
- Feature extraction:
  + TF-IDF.
  + Word2Vec (Glove embedding).
  + Bag-of-Words.
- Shallow learning algorithms:
  + Linear SVM.
  + Logistic Regressor.
  + Naive Bayes.
  + Random Forest.
- Language model-based algorithms, for example:
  + distilbert-base-uncased.
  + bert-base-uncased.
  + roberta-base.
  + ... any other language model which is available on HuggingFace (https://huggingface.co/models).
