import pandas as pd
import numpy as np
import re
from datetime import datetime

# ==========================================
# LOAD RAW DATA
# ==========================================

def load_data():
    df = pd.read_csv("data/tickets.csv")
    return df


# ==========================================
# TEXT CLEANING FUNCTION (NLP PREPROCESSING)
# ==========================================

def clean_text(text):
    if pd.isnull(text):
        return ""

    text = str(text).lower()

    # remove special characters
    text = re.sub(r"[^a-z0-9\s]", "", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==========================================
# PRIORITY MAPPING (IF NEEDED FOR BALANCE)
# ==========================================

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


# ==========================================
# MAIN PREPROCESSING PIPELINE
# ==========================================

def create_dataset():
    df = load_data()

    # ------------------------------------------
    # TEXT FEATURES
    # ------------------------------------------
    df["Ticket Description Clean"] = df["Ticket Description"].apply(clean_text)
    df["Ticket Subject Clean"] = df["Ticket Subject"].apply(clean_text)

    # Combine text for NLP model
    df["Text"] = (
        df["Ticket Subject Clean"] + " " + df["Ticket Description Clean"]
    )

    # ------------------------------------------
    # TARGET 1: CATEGORY (Ticket Type)
    # ------------------------------------------
    df["Category"] = df["Ticket Type"]

    # ------------------------------------------
    # TARGET 2: PRIORITY
    # ------------------------------------------
    df["Priority"] = df["Ticket Priority"].apply(normalize_priority)

    # ------------------------------------------
    # TIME FEATURES
    # ------------------------------------------
    df["Date of Purchase"] = pd.to_datetime(df["Date of Purchase"])

    df["Year"] = df["Date of Purchase"].dt.year
    df["Month"] = df["Date of Purchase"].dt.month
    df["Day"] = df["Date of Purchase"].dt.day
    df["Weekday"] = df["Date of Purchase"].dt.weekday

    # ------------------------------------------
    # CUSTOMER FEATURES (OPTIONAL ENRICHMENT)
    # ------------------------------------------
    df["Customer Age Group"] = pd.cut(
        df["Customer Age"],
        bins=[0, 18, 30, 45, 60, 100],
        labels=["Teen", "Young", "Adult", "MidAge", "Senior"]
    )

    # ------------------------------------------
    # DROP UNNECESSARY COLUMNS FOR MODEL TRAINING
    # ------------------------------------------
    drop_cols = [
        "Ticket ID",
        "Customer Name",
        "Customer Email",
        "Ticket Subject Clean",
        "Ticket Description Clean"
    ]

    df_final = df.drop(columns=drop_cols, errors="ignore")

    # ------------------------------------------
    # SAVE PROCESSED DATASET
    # ------------------------------------------
    df_final.to_csv("data/processed_tickets.csv", index=False)

    print("✅ Dataset created successfully!")
    print("Saved at: data/processed_tickets.csv")


# ==========================================
# RUN SCRIPT
# ==========================================

if __name__ == "__main__":
    create_dataset()