# Sentiment Analysis Tutorial

## Setup environment
```
conda create -p .env/ python=3.10
pip install -r requirements.txt
```

## Training and Evaluate

See the notebook TrainingAndEvaluate.ipynb.

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

## Google Colab Version
Instead of setting up the repository in your local machine, you can also use the code on Google Colab, see [Demo](https://colab.research.google.com/drive/14gIeVL9gcO7MdxMbmH5JZfSNwfeR5cjj?usp=sharing).

Training RoBERTa model for Sentiment Analysis: https://colab.research.google.com/drive/1ZNe56SRCHTm5iBOusZKSjAN0ZLT60Udw?usp=sharing
