import os
from pprint import pp

import pandas as pd
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix
from  sklearn.metrics import classification_report

analyzer = SentimentIntensityAnalyzer()

CUSTOMER_COMPLAINT = "this product is not good and I am not happy with it"
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
    return 1 if scores["pos"] > 0 else 0


# if not os.path.exists(DATASET_PREPROCESS_PATH):
#     df = pd.read_csv(DATASET_PATH)
#     df["reviewText"] = df["reviewText"].apply(preprocess)
#     df.to_csv(DATASET_PREPROCESS_PATH, index=False)

# if not os.path.exists(DATASET_SENTIMENT_PATH):
#     df = pd.read_csv(DATASET_PREPROCESS_PATH)
#     df["reviewText"] = df["reviewText"].astype(str)
#     df["sentiment"] = df["reviewText"].apply(get_sentiment)
#     df.to_csv(DATASET_SENTIMENT_PATH, index=False)

# df = pd.read_csv(DATASET_SENTIMENT_PATH)
# pp(df)
# # pp(preprocess(customer_complaint))
# print(confusion_matrix(df['Positive'], df['sentiment']))
# print(classification_report(df['Positive'], df['sentiment']))

pp(analyzer.polarity_scores(preprocess(CUSTOMER_COMPLAINT)))