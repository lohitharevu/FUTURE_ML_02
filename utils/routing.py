import pandas as pd

from predictor import assign_team, get_sla


# ==========================================
# ROUTE SINGLE TICKET
# ==========================================

def route_ticket(ticket_text, category, priority):

    team = assign_team(category)
    sla = get_sla(priority)

    return {
        "assigned_team": team,
        "response_sla": sla
    }


# ==========================================
# ESCALATION LOGIC
# ==========================================

def check_escalation(priority, sentiment_score=None):

    """
    Simple business rule:
    - High priority always escalates
    - Medium + negative sentiment escalates
    """

    if priority == "High":
        return True

    if priority == "Medium" and sentiment_score is not None:
        if sentiment_score < 0.3:
            return True

    return False


# ==========================================
# TEAM LOAD DISTRIBUTION
# ==========================================

def team_workload(df):

    """
    Simulates workload distribution across teams
    """

    if "category" not in df.columns:
        return {}

    df["assigned_team"] = df["category"].apply(assign_team)

    workload = df["assigned_team"].value_counts().reset_index()
    workload.columns = ["Team", "Ticket Count"]

    return workload


# ==========================================
# PRIORITY BASED QUEUE ORDERING
# ==========================================

def prioritize_queue(df):

    priority_map = {
        "High": 1,
        "Medium": 2,
        "Low": 3
    }

    df["priority_rank"] = df["priority"].map(priority_map)

    sorted_df = df.sort_values(by="priority_rank", ascending=True)

    return sorted_df


# ==========================================
# AUTO ROUTING SUMMARY
# ==========================================

def routing_summary(df):

    workload = team_workload(df)

    high_count = len(df[df["priority"] == "High"])
    medium_count = len(df[df["priority"] == "Medium"])
    low_count = len(df[df["priority"] == "Low"])

    summary = {

        "total_tickets": len(df),

        "high_priority": high_count,
        "medium_priority": medium_count,
        "low_priority": low_count,

        "teams_active": len(workload) if isinstance(workload, pd.DataFrame) else 0
    }

    return summary