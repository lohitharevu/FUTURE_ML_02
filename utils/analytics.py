import pandas as pd

# ==========================================
# LOAD DATA
# ==========================================

def load_data():
    df = pd.read_csv("data/processed_tickets.csv")
    return df


# ==========================================
# KPI METRICS
# ==========================================

def get_kpis():

    df = load_data()

    total_tickets = len(df)

    high_priority = len(df[df["priority"] == "High"])
    medium_priority = len(df[df["priority"] == "Medium"])
    low_priority = len(df[df["priority"] == "Low"])

    avg_satisfaction = df["Customer Satisfaction Rating"].mean()

    return {
        "total_tickets": total_tickets,
        "high_priority": high_priority,
        "medium_priority": medium_priority,
        "low_priority": low_priority,
        "avg_satisfaction": round(avg_satisfaction, 2)
    }


# ==========================================
# CATEGORY DISTRIBUTION
# ==========================================

def category_distribution():

    df = load_data()

    category = (
        df["category"]
        .value_counts()
        .reset_index()
    )

    category.columns = ["Category", "Count"]

    return category


# ==========================================
# PRIORITY DISTRIBUTION
# ==========================================

def priority_distribution():

    df = load_data()

    priority = (
        df["priority"]
        .value_counts()
        .reset_index()
    )

    priority.columns = ["Priority", "Count"]

    return priority


# ==========================================
# TICKET CHANNEL ANALYSIS
# ==========================================

def channel_analysis():

    df = load_data()

    channel = (
        df["Ticket Channel"]
        .value_counts()
        .reset_index()
    )

    channel.columns = ["Channel", "Count"]

    return channel


# ==========================================
# MONTHLY TICKET TRENDS
# ==========================================

def monthly_trends():

    df = load_data()

    df["Date of Purchase"] = pd.to_datetime(df["Date of Purchase"])

    monthly = (
        df.groupby(df["Date of Purchase"].dt.to_period("M"))
        .size()
        .reset_index(name="Ticket Count")
    )

    monthly["Date of Purchase"] = monthly["Date of Purchase"].astype(str)

    return monthly


# ==========================================
# PRODUCT WISE ANALYSIS
# ==========================================

def product_analysis():

    df = load_data()

    products = (
        df["Product Purchased"]
        .value_counts()
        .reset_index()
    )

    products.columns = ["Product", "Tickets"]

    return products


# ==========================================
# SLA PERFORMANCE
# ==========================================

def sla_analysis():

    df = load_data()

    # convert time columns if needed
    df["Time to Resolution"] = pd.to_numeric(
        df["Time to Resolution"],
        errors="coerce"
    )

    avg_resolution_time = df["Time to Resolution"].mean()

    return {
        "avg_resolution_time": round(avg_resolution_time, 2)
    }


# ==========================================
# SUPPORT HEALTH SCORE
# ==========================================

def support_health_score():

    df = load_data()

    high_ratio = len(df[df["priority"] == "High"]) / len(df)

    satisfaction = df["Customer Satisfaction Rating"].mean()

    score = (
        (1 - high_ratio) * 50 +
        (satisfaction / 5) * 50
    )

    return round(score, 2)