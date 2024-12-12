from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Carregar o dataset
df = pd.read_csv(DATASET_PREPROCESS_PATH)

# Vetorização
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['complaint_text'])
y = df['sentiment']

# Divisão de dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinamento do modelo
model = MultinomialNB()
model.fit(X_train, y_train)

# Predição e avaliação
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))




new_text = "This product is awful, I want a refund!"
processed_text = preprocess(new_text)
vectorized_text = vectorizer.transform([processed_text])
predicted_sentiment = model.predict(vectorized_text)
print(f"Sentimento previsto: {predicted_sentiment}")
