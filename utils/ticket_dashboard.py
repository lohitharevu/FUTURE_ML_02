import pandas as pd

from analytics import (
    get_kpis,
    category_distribution,
    priority_distribution,
    channel_analysis,
    monthly_trends,
    product_analysis,
    sla_analysis,
    support_health_score
)

from insights import executive_summary


# ==========================================
# DASHBOARD METRICS (MAIN OVERVIEW)
# ==========================================

def dashboard_metrics():

    kpis = get_kpis()
    sla = sla_analysis()
    health = support_health_score()

    return {
        "kpis": kpis,
        "sla": sla,
        "health_score": health
    }


# ==========================================
# CHART DATA PROVIDERS
# ==========================================

def chart_data():

    return {

        "category_distribution": category_distribution(),
        "priority_distribution": priority_distribution(),
        "channel_distribution": channel_analysis(),
        "monthly_trends": monthly_trends(),
        "product_analysis": product_analysis()
    }


# ==========================================
# EXECUTIVE INSIGHTS
# ==========================================

def ai_insights():

    return executive_summary()


# ==========================================
# FULL DASHBOARD DATA PACKAGE
# ==========================================

def get_dashboard_data():

    return {

        "metrics": dashboard_metrics(),

        "charts": chart_data(),

        "insights": ai_insights()
    }


# ==========================================
# FILTERED DASHBOARD VIEW (OPTIONAL)
# ==========================================

def filtered_dashboard(df, priority=None, category=None):

    if priority:
        df = df[df["priority"] == priority]

    if category:
        df = df[df["category"] == category]

    return {

        "filtered_kpis": {
            "total_tickets": len(df),
            "high_priority": len(df[df["priority"] == "High"]),
            "medium_priority": len(df[df["priority"] == "Medium"]),
            "low_priority": len(df[df["priority"] == "Low"])
        },

        "filtered_category": df["category"].value_counts().reset_index(),

        "filtered_priority": df["priority"].value_counts().reset_index()
    }