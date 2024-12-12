import os
from pprint import pp

import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

analyzer = SentimentIntensityAnalyzer()

CUSTOMER_COMPLAINT = "this product is normal"
DATASET_PATH = "dataset/customer_complaints.csv"
DATASET_PREPROCESS_PATH = "dataset/customer_complaints_preprocess.csv"
DATASET_SENTIMENT_PATH = "dataset/customer_complaints_sentiment.csv"


def preprocess(text: str):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [
        token for token in tokens if token not in stopwords.words("english")
    ]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return " ".join(lemmatized_tokens)


def get_sentiment(text: str):
    scores = analyzer.polarity_scores(text)
    if scores["compound"] >= 0.05:
        return 1
    if scores["compound"] <= -0.05:
        return -1
    return 0


pp(get_sentiment(preprocess(CUSTOMER_COMPLAINT)))
