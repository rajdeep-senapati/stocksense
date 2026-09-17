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
    return pd.read_csv(
        "data/processed/stock_inventory_decisions.csv", dtype={"StockCode": str}
    )


@st.cache_data
def load_forecast_data():
    return pd.read_csv(
        "data/processed/future_7_day_forecast.csv", dtype={"StockCode": str}
    )


@st.cache_data
def load_demand_history():
    return pd.read_csv(
        "data/processed/production_demand_history.csv", dtype={"StockCode": str}
    )


decision_data = load_decision_data()
forecast_data = load_forecast_data()
demand_history = load_demand_history()

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

    # 7-Day Demand Forecast
    st.subheader("7-Day Demand Forecast")

    sku_forecast = forecast_data[
        forecast_data["StockCode"].astype(str) == selected_sku
    ].copy()

    sku_forecast["Date"] = pd.to_datetime(sku_forecast["Date"])

    sku_forecast = sku_forecast.sort_values("Date")

    chart_data = sku_forecast.set_index("Date")[["Predicted_Demand"]]

    st.line_chart(chart_data, width="stretch")

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

    st.caption(
        "Inventory is simulated from recent historical demand because "
        "the source dataset does not contain actual on-hand inventory."
    )

# -----------------------------
# Forecast
# -----------------------------
elif page == "Forecast":

    st.header("7-Day Demand Forecast")

    selected_sku = st.selectbox(
        "Select SKU", sorted(forecast_data["StockCode"].astype(str).unique())
    )

    sku_forecast = forecast_data[
        forecast_data["StockCode"].astype(str) == selected_sku
    ].copy()

    sku_forecast["Date"] = pd.to_datetime(sku_forecast["Date"])
    sku_forecast = sku_forecast.sort_values("Date")

    total_forecast = sku_forecast["Predicted_Demand"].sum()
    average_daily_forecast = sku_forecast["Predicted_Demand"].mean()

    col1, col2 = st.columns(2)

    col1.metric("7-Day Forecast", f"{total_forecast:.2f} units")

    col2.metric("Average Daily Forecast", f"{average_daily_forecast:.2f} units")

    st.divider()

    st.subheader(f"Recent Demand vs Forecast — SKU {selected_sku}")

    sku_actual = demand_history[
        demand_history["StockCode"].astype(str) == selected_sku
    ].copy()

    sku_actual["Date"] = pd.to_datetime(sku_actual["Date"])

    sku_actual = (
        sku_actual.sort_values("Date")
        .tail(14)[["Date", "Demand"]]
        .rename(columns={"Demand": "Actual Demand"})
    )

    forecast_chart = sku_forecast[["Date", "Predicted_Demand"]].rename(
        columns={"Predicted_Demand": "Forecast Demand"}
    )

    combined_chart = pd.concat(
        [sku_actual.set_index("Date"), forecast_chart.set_index("Date")], axis=1
    )

    st.line_chart(combined_chart, width="stretch")

    st.caption(
        "The chart shows the last 14 days of recorded demand followed by "
        "the next 7 days of model forecasts."
    )

    st.divider()

    st.subheader("Daily Forecast")

    display_forecast = sku_forecast.copy()

    display_forecast["Date"] = display_forecast["Date"].dt.strftime("%Y-%m-%d")

    display_forecast = display_forecast.rename(
        columns={"StockCode": "SKU", "Predicted_Demand": "Predicted Demand"}
    )

    st.dataframe(
        display_forecast[["SKU", "Date", "Predicted Demand"]],
        hide_index=True,
        width="stretch",
    )

# -----------------------------
# Inventory Risk
# -----------------------------
elif page == "Inventory Risk":

    st.header("Inventory Risk Dashboard")

    st.markdown(
        "Identify SKUs where simulated inventory falls below "
        "the forecast-based reorder point."
    )

    # Risk filter
    risk_options = ["All", "Healthy", "Watch", "High", "Critical"]

    selected_risk = st.selectbox("Filter by Risk Level", risk_options)

    if selected_risk == "All":
        filtered_data = decision_data.copy()
    else:
        filtered_data = decision_data[
            decision_data["risk_level"] == selected_risk
        ].copy()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("SKUs", f"{len(filtered_data):,}")

    col2.metric(
        "Average Inventory", f"{filtered_data['simulated_inventory'].mean():.1f}"
    )

    col3.metric(
        "Average Reorder Point", f"{filtered_data['forecast_reorder_point'].mean():.1f}"
    )

    col4.metric(
        "Average Inventory Gap", f"{filtered_data['inventory_gap_pct'].mean():.1f}%"
    )

    st.divider()

    # Risk distribution
    st.subheader("Risk Distribution")

    risk_counts = (
        decision_data["risk_level"]
        .value_counts()
        .reindex(["Healthy", "Watch", "High", "Critical"], fill_value=0)
    )

    st.bar_chart(risk_counts, width="stretch")

    st.divider()

    # Inventory vs Reorder Point
    st.subheader("Top 15 SKUs by Inventory Gap")

    chart_data = (
        filtered_data.sort_values("inventory_gap_pct", ascending=False).head(15).copy()
    )

    chart_data["StockCode"] = chart_data["StockCode"].astype(str)

    chart_data = chart_data[
        ["StockCode", "simulated_inventory", "forecast_reorder_point"]
    ].set_index("StockCode")

    chart_data = chart_data.rename(
        columns={
            "simulated_inventory": "Current Inventory",
            "forecast_reorder_point": "Reorder Point",
        }
    )

    st.bar_chart(chart_data, width="stretch")

    # Detailed table
    st.subheader("Priority SKUs")

    priority_data = (
        filtered_data.sort_values("inventory_gap_pct", ascending=False).head(10).copy()
    )

    priority_data["StockCode"] = priority_data["StockCode"].astype(str)

    priority_data = priority_data[
        [
            "StockCode",
            "inventory_gap_pct",
            "forecast_reorder_point",
            "simulated_inventory",
            "risk_level",
            "recommended_action",
        ]
    ].rename(
        columns={
            "StockCode": "SKU",
            "inventory_gap_pct": "Gap %",
            "forecast_reorder_point": "Reorder Point",
            "simulated_inventory": "Current Inventory",
            "risk_level": "Risk Level",
            "recommended_action": "Recommended Action",
        }
    )

    st.dataframe(priority_data, hide_index=True, width="stretch")

    st.divider()

    st.subheader("Risk Summary")

    risk_summary = (
        decision_data["risk_level"]
        .value_counts()
        .reindex(["Healthy", "Watch", "High", "Critical"], fill_value=0)
        .reset_index()
    )

    risk_summary.columns = ["Risk Level", "SKU Count"]

    risk_summary["Share"] = (
        risk_summary["SKU Count"] / len(decision_data) * 100
    ).round(1)

    st.dataframe(risk_summary, hide_index=True, width="stretch")


# -----------------------------
# Recommended Actions
# -----------------------------

elif page == "Recommended Actions":

    st.header("Recommended Actions")

    st.markdown(
        "Prioritize inventory decisions based on forecast demand, "
        "reorder point, and simulated inventory position."
    )

    # Action filter
    action_options = [
        "All",
        "Reorder immediately",
        "Reorder soon",
        "Monitor closely",
        "No action",
    ]

    selected_action = st.selectbox("Filter by Recommended Action", action_options)

    if selected_action == "All":
        action_data = decision_data.copy()
    else:
        action_data = decision_data[
            decision_data["recommended_action"] == selected_action
        ].copy()

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    col1.metric("SKUs", f"{len(action_data):,}")

    col2.metric(
        "Total Inventory Gap",
        f"{action_data['inventory_gap'].clip(lower=0).sum():,.0f}",
    )

    col3.metric(
        "Average Gap %", f"{action_data['inventory_gap_pct'].clip(lower=0).mean():.1f}%"
    )

    st.divider()

    # Priority table
    st.subheader("Priority Actions")

    priority_data = action_data.sort_values("inventory_gap_pct", ascending=False).copy()

    priority_data["StockCode"] = priority_data["StockCode"].astype(str)

    priority_data = priority_data[
        [
            "StockCode",
            "forecast_7_day_demand",
            "simulated_inventory",
            "forecast_reorder_point",
            "inventory_gap",
            "inventory_gap_pct",
            "risk_level",
            "recommended_action",
        ]
    ].rename(
        columns={
            "StockCode": "SKU",
            "forecast_7_day_demand": "7-Day Forecast",
            "simulated_inventory": "Current Inventory",
            "forecast_reorder_point": "Reorder Point",
            "inventory_gap": "Inventory Gap",
            "inventory_gap_pct": "Gap %",
            "risk_level": "Risk Level",
            "recommended_action": "Recommended Action",
        }
    )

    st.dataframe(priority_data, hide_index=True, width="stretch")

    st.divider()

    st.subheader("Action Distribution")

    action_counts = (
        decision_data["recommended_action"]
        .value_counts()
        .reindex(
            ["Reorder immediately", "Reorder soon", "Monitor closely", "No action"],
            fill_value=0,
        )
    )

    st.bar_chart(action_counts, width="stretch")

    st.caption(
        "Recommendations are based on forecast-derived reorder points and "
        "simulated inventory. They are decision-support outputs, not actual purchase orders."
    )
