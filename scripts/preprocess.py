import pandas as pd
import re

def load_data():
    df = pd.read_csv("data/tickets.csv")
    return df

def clean_text(text):
    if pd.isnull(text):
        return ""

    text = str(text).lower()

    # remove special characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

def normalize_priority(priority):
    if pd.isnull(priority):
        return "Low"

    priority = str(priority).lower()

    if "high" in priority:
        return "High"
    elif "medium" in priority:
        return "Medium"
    else:
        return "Low"

def preprocess():

    df = load_data()
    df["clean_subject"] = df["Ticket Subject"].apply(clean_text)
    df["clean_description"] = df["Ticket Description"].apply(clean_text)

    # Combine text for NLP model
    df["text"] = df["clean_subject"] + " " + df["clean_description"]

    df["category"] = df["Ticket Type"]
    df["priority"] = df["Ticket Priority"].apply(normalize_priority)

    df["Date of Purchase"] = pd.to_datetime(df["Date of Purchase"])

    df["year"] = df["Date of Purchase"].dt.year
    df["month"] = df["Date of Purchase"].dt.month
    df["day"] = df["Date of Purchase"].dt.day
    df["weekday"] = df["Date of Purchase"].dt.weekday
    df["age_group"] = pd.cut(
        df["Customer Age"],
        bins=[0, 18, 30, 45, 60, 100],
        labels=["Teen", "Young", "Adult", "MidAge", "Senior"]
    )

    drop_cols = [
        "Ticket ID",
        "Customer Name",
        "Customer Email"
    ]

    df = df.drop(columns=drop_cols, errors="ignore")
    df.to_csv("data/processed_tickets.csv", index=False)

    print("✅ Preprocessing completed!")
    print("Saved: data/processed_tickets.csv")

if __name__ == "__main__":
    preprocess()