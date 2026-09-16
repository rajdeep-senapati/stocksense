import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="StockSense", page_icon="📦", layout="wide")


# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_decision_data():
    return pd.read_csv("data/processed/stock_inventory_decisions.csv")


decision_data = load_decision_data()


# -----------------------------
# Header
# -----------------------------
st.title("📦 StockSense")
st.subheader("Predictive Inventory Intelligence & Decision Support")

st.markdown(
    "Forecast demand, identify inventory risk, and prioritize replenishment decisions."
)


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Executive Overview",
        "SKU Explorer",
        "Forecast",
        "Inventory Risk",
        "Recommended Actions",
    ],
)


# -----------------------------
# Executive Overview
# -----------------------------
if page == "Executive Overview":

    st.header("Executive Overview")

    total_skus = decision_data["StockCode"].nunique()

    total_reorder = (
        decision_data["recommended_action"]
        .isin(["Reorder immediately", "Reorder soon"])
        .sum()
    )

    critical = (decision_data["risk_level"] == "Critical").sum()

    healthy = (decision_data["risk_level"] == "Healthy").sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("SKUs Analyzed", f"{total_skus:,}")

    col2.metric("Reorder Required", f"{total_reorder:,}")

    col3.metric("Critical Risk", f"{critical:,}")

    col4.metric("Healthy", f"{healthy:,}")

    st.divider()

    st.subheader("Inventory Risk Distribution")

    risk_counts = (
        decision_data["risk_level"]
        .value_counts()
        .reindex(["Healthy", "Watch", "High", "Critical"], fill_value=0)
    )

    st.bar_chart(risk_counts, width="stretch")

    st.subheader("Recommended Actions")

    action_counts = decision_data["recommended_action"].value_counts()

    st.bar_chart(action_counts, width="stretch")


# -----------------------------
# SKU Explorer
# -----------------------------
elif page == "SKU Explorer":

    st.header("SKU Explorer")

    selected_sku = st.selectbox(
        "Select SKU", sorted(decision_data["StockCode"].astype(str).unique())
    )

    sku_data = decision_data[
        decision_data["StockCode"].astype(str) == selected_sku
    ].iloc[0]

    st.subheader(f"SKU: {selected_sku}")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Mean Daily Demand", f"{sku_data['mean_daily_demand']:.2f}")

    col2.metric("7-Day Forecast", f"{sku_data['forecast_7_day_demand']:.2f}")

    col3.metric("Safety Stock", f"{sku_data['safety_stock']:.2f}")

    col4.metric("Simulated Inventory", f"{sku_data['simulated_inventory']:.0f}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Inventory Position")

        inventory_metrics = pd.DataFrame(
            {
                "Metric": [
                    "Forecast 7-Day Demand",
                    "Safety Stock",
                    "Reorder Point",
                    "Simulated Inventory",
                ],
                "Value": [
                    sku_data["forecast_7_day_demand"],
                    sku_data["safety_stock"],
                    sku_data["forecast_reorder_point"],
                    sku_data["simulated_inventory"],
                ],
            }
        )

        st.dataframe(inventory_metrics, hide_index=True, width="stretch")

    with col2:

        st.subheader("Risk Assessment")

        st.metric("Risk Level", sku_data["risk_level"])

        st.metric("Inventory Gap", f"{sku_data['inventory_gap']:.2f}")

        st.metric("Inventory Gap %", f"{sku_data['inventory_gap_pct']:.2f}%")

        st.info(f"Recommended Action: {sku_data['recommended_action']}")
