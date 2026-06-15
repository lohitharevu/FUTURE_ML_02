import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

print("Loading dataset...")

df = pd.read_csv("data/processed_tickets.csv")

X = df["text"]
y = df["priority"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Vectorizing text...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Training Priority Model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

acc = accuracy_score(y_test, y_pred)

print("\n========== PRIORITY MODEL RESULTS ==========")
print(f"Accuracy: {acc:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

joblib.dump(model, "models/priority_model.pkl")
joblib.dump(vectorizer, "models/priority_vectorizer.pkl")

print("\n✅ Priority model saved successfully!")
print("Saved: models/priority_model.pkl")
print("Saved: models/priority_vectorizer.pkl")