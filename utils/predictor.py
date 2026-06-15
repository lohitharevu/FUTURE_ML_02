import joblib
import numpy as np

category_model = joblib.load("models/category_model.pkl")
priority_model = joblib.load("models/priority_model.pkl")

category_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
priority_vectorizer = joblib.load("models/priority_vectorizer.pkl")
def clean_input(text):
    if not text:
        return ""

    return text.lower().strip()

def predict_category(text):

    text = clean_input(text)

    vec = category_vectorizer.transform([text])

    prediction = category_model.predict(vec)[0]

    return prediction

def predict_priority(text):

    text = clean_input(text)

    vec = priority_vectorizer.transform([text])

    prediction = priority_model.predict(vec)[0]

    return prediction

def predict_ticket(text):

    category = predict_category(text)
    priority = predict_priority(text)

    # Confidence (approximation using model decision score if available)
    confidence = None

    try:
        # LinearSVC does not always support predict_proba
        scores = category_model.decision_function(
            category_vectorizer.transform([text])
        )

        confidence = float(np.max(scores))

    except:
        confidence = 0.0

    return {
        "category": category,
        "priority": priority,
        "confidence_score": round(confidence, 2)
    }

def get_sla(priority):

    if priority == "High":
        return "2 Hours"

    elif priority == "Medium":
        return "8 Hours"

    else:
        return "24 Hours"

def assign_team(category):

    mapping = {
        "Billing": "Finance Team",
        "Technical Issue": "Engineering Team",
        "Account Issue": "Customer Success Team",
        "Refund Request": "Finance Team",
        "Complaint": "Support Escalation Team",
        "General Query": "General Support Team"
    }

    return mapping.get(category, "Support Team")