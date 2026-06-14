import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# LOAD DATA
# ==========================================

print("Loading dataset...")

df = pd.read_csv("data/processed_tickets.csv")

# ==========================================
# FEATURES & TARGET
# ==========================================

X = df["text"]
y = df["category"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# TF-IDF VECTORIZER
# ==========================================

print("Vectorizing text...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==========================================
# MODEL TRAINING
# ==========================================

print("Training Category Model...")

model = LinearSVC()

model.fit(X_train_vec, y_train)

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test_vec)

# ==========================================
# EVALUATION
# ==========================================

acc = accuracy_score(y_test, y_pred)

print("\n========== CATEGORY MODEL RESULTS ==========")
print(f"Accuracy: {acc:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==========================================
# SAVE MODEL + VECTORIZER
# ==========================================

joblib.dump(model, "models/category_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\n✅ Model saved successfully!")
print("Saved: models/category_model.pkl")
print("Saved: models/tfidf_vectorizer.pkl")