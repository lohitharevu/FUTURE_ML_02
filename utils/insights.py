import pandas as pd

def load_data():
    df = pd.read_csv("data/processed_tickets.csv")
    return df

def ticket_insights():

    df = load_data()

    total = len(df)

    high = len(df[df["priority"] == "High"])
    medium = len(df[df["priority"] == "Medium"])
    low = len(df[df["priority"] == "Low"])

    insights = []

    insights.append(f"Total support tickets received: {total}.")
    insights.append(f"High priority tickets: {high}.")
    insights.append(f"Medium priority tickets: {medium}.")
    insights.append(f"Low priority tickets: {low}.")

    if high / total > 0.3:
        insights.append("⚠️ High number of urgent tickets detected. Support team is under pressure.")
    else:
        insights.append("Support ticket urgency is within manageable levels.")

    return insights

def category_insights():

    df = load_data()

    category_counts = df["category"].value_counts()

    top_category = category_counts.index[0]
    top_count = category_counts.iloc[0]

    insights = []

    insights.append(f"Most common issue type is '{top_category}' with {top_count} tickets.")

    if top_count / len(df) > 0.4:
        insights.append(f"{top_category} dominates support requests and needs process optimization.")

    insights.append("Category distribution helps identify recurring customer problems.")

    return insights

def product_insights():

    df = load_data()

    product_counts = df["Product Purchased"].value_counts()

    top_product = product_counts.index[0]
    top_count = product_counts.iloc[0]

    insights = []

    insights.append(f"Most problematic product is '{top_product}' with {top_count} tickets.")

    insights.append("High ticket volume for a product may indicate quality or usability issues.")

    return insights

def customer_experience_insights():

    df = load_data()

    avg_rating = df["Customer Satisfaction Rating"].mean()

    insights = []

    insights.append(f"Average customer satisfaction rating is {round(avg_rating, 2)} out of 5.")

    if avg_rating < 3:
        insights.append("⚠️ Customer satisfaction is low. Immediate service improvements required.")
    elif avg_rating < 4:
        insights.append("Customer satisfaction is moderate. There is room for improvement.")
    else:
        insights.append("Customer satisfaction is strong.")

    return insights

def sla_insights():

    df = load_data()

    df["Time to Resolution"] = pd.to_numeric(df["Time to Resolution"], errors="coerce")

    avg_resolution = df["Time to Resolution"].mean()

    insights = []

    insights.append(f"Average resolution time is {round(avg_resolution, 2)} units.")

    if avg_resolution > 48:
        insights.append("⚠️ Resolution time is high. Support efficiency needs improvement.")
    else:
        insights.append("Resolution time is within acceptable range.")

    return insights

def executive_summary():

    summary = []

    summary.extend(ticket_insights())
    summary.extend(category_insights())
    summary.extend(product_insights())
    summary.extend(customer_experience_insights())
    summary.extend(sla_insights())

    return summary