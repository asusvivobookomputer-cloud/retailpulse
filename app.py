"""Interactive dashboard.   Run:  streamlit run app.py   (after `python main.py`)"""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from retailpulse import analysis, config

st.set_page_config(page_title="RetailPulse Analytics", page_icon="📊", layout="wide")

NAVY, ORANGE = "#1F4E79", "#E07B39"


# ----------------------------------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading data from SQLite ...")
def load_lines() -> pd.DataFrame:
    conn = analysis.connect()
    try:
        return analysis.load_lines(conn)
    finally:
        conn.close()


@st.cache_data
def load_table(name: str) -> pd.DataFrame:
    return pd.read_csv(config.TABLES_DIR / name)


if not config.DB_PATH.exists():
    st.error("Database not found. Run `python main.py` first to build the data and outputs.")
    st.stop()

lines_all = load_lines()

# --------------------------------------------------------------------------------------- sidebar
st.sidebar.title("Filters")
d_min, d_max = lines_all["order_date"].min().date(), lines_all["order_date"].max().date()
date_range = st.sidebar.date_input("Order date", (d_min, d_max), min_value=d_min, max_value=d_max)
countries = st.sidebar.multiselect("Country", sorted(lines_all["country"].unique()))
categories = st.sidebar.multiselect("Category", sorted(lines_all["category"].unique()))
channels = st.sidebar.multiselect("Sales channel", sorted(lines_all["sales_channel"].unique()))

lines = lines_all
if isinstance(date_range, tuple) and len(date_range) == 2:
    lines = lines[(lines["order_date"].dt.date >= date_range[0]) & (lines["order_date"].dt.date <= date_range[1])]
for col, sel in (("country", countries), ("category", categories), ("sales_channel", channels)):
    if sel:
        lines = lines[lines[col].isin(sel)]

if lines.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# ------------------------------------------------------------------------------------------ header
st.title("📊 RetailPulse Analytics")
st.caption("SQL · pandas · scikit-learn · forecasting — filters on the left apply to the live tabs "
           "(Overview, Customers, Retention, Products).")

k = analysis.kpis(lines)
c = st.columns(6)
c[0].metric("Net revenue", f"${k['net_revenue']:,.0f}")
c[1].metric("Net profit", f"${k['net_profit']:,.0f}")
c[2].metric("Orders", f"{k['orders']:,}")
c[3].metric("Customers", f"{k['customers']:,}")
c[4].metric("AOV", f"${k['aov']:,.2f}")
c[5].metric("Gross margin", f"{k['gross_margin_pct']:.1%}")

tabs = st.tabs(["Overview", "Customers (RFM)", "Retention", "Products", "Churn risk", "Forecast", "Marketing"])

# ---------------------------------------------------------------------------------------- overview
with tabs[0]:
    monthly = analysis.monthly_sales(lines)
    fig = go.Figure()
    fig.add_bar(x=monthly["order_month"], y=monthly["revenue"], name="Revenue", marker_color=NAVY)
    fig.add_scatter(x=monthly["order_month"], y=monthly["profit"], name="Profit", line=dict(color=ORANGE, width=3))
    fig.update_layout(title="Monthly revenue and profit", height=380, legend=dict(orientation="h"),
                      margin=dict(t=50, b=10))
    st.plotly_chart(fig, width="stretch")
    a, b = st.columns(2)
    cat = analysis.category_performance(lines)
    a.plotly_chart(px.bar(cat, x="revenue", y="category", orientation="h", title="Revenue by category",
                          color_discrete_sequence=[NAVY]).update_layout(yaxis=dict(autorange="reversed")),
                   width="stretch")
    geo = lines.groupby("country")["net_revenue"].sum().sort_values().reset_index()
    b.plotly_chart(px.bar(geo, x="net_revenue", y="country", orientation="h", title="Revenue by country",
                          color_discrete_sequence=[ORANGE]), width="stretch")
    st.dataframe(cat.style.format({"revenue": "${:,.0f}", "profit": "${:,.0f}", "margin_pct": "{:.1%}",
                                   "return_rate_pct": "{:.1%}", "revenue_share_pct": "{:.1%}"}),
                 width="stretch", hide_index=True)

# --------------------------------------------------------------------------------------------- RFM
with tabs[1]:
    rfm_df = analysis.rfm(lines)
    summ = analysis.rfm_segment_summary(rfm_df)
    a, b = st.columns([3, 2])
    a.plotly_chart(px.treemap(summ, path=["segment"], values="revenue", color="avg_recency_days",
                              color_continuous_scale="Blues_r", title="Revenue by RFM segment (colour = avg recency)"),
                   width="stretch")
    b.dataframe(summ[["segment", "customers", "revenue", "avg_frequency"]].style.format(
        {"revenue": "${:,.0f}", "avg_frequency": "{:.1f}"}), hide_index=True, width="stretch")
    seg = st.selectbox("Inspect a segment", summ["segment"])
    out = rfm_df[rfm_df["segment"] == seg].sort_values("monetary", ascending=False)
    st.dataframe(out, hide_index=True, width="stretch")
    st.download_button("Download segment as CSV", out.to_csv(index=False), f"rfm_{seg}.csv", "text/csv")

# --------------------------------------------------------------------------------------- retention
with tabs[2]:
    coh = analysis.cohort_retention(lines).drop(columns="cohort_size")
    fig = px.imshow(coh, text_auto=".0%", color_continuous_scale="Blues", zmin=0, zmax=0.5, aspect="auto",
                    labels=dict(x="Months since first purchase", y="Cohort", color="Retention"),
                    title="Cohort retention heatmap")
    fig.update_layout(height=650)
    st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------------------- products
with tabs[3]:
    par = analysis.pareto_products(lines)
    fig = go.Figure()
    fig.add_bar(x=par["rank"], y=par["revenue"], marker_color=par["abc_class"].map(
        {"A": NAVY, "B": ORANGE, "C": "#999999"}), name="Revenue")
    fig.add_scatter(x=par["rank"], y=par["cum_revenue_share"], yaxis="y2", name="Cumulative share",
                    line=dict(color="#C0392B"))
    fig.update_layout(title="Pareto: product revenue (A / B / C)", height=400,
                      yaxis2=dict(overlaying="y", side="right", tickformat=".0%", range=[0, 1.05]),
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    st.subheader("Products frequently bought together")
    st.dataframe(analysis.market_basket(lines), hide_index=True, width="stretch")

# -------------------------------------------------------------------------------------- churn risk
with tabs[4]:
    try:
        scores = load_table("churn_scores.csv")
        metrics = json.loads((config.OUTPUTS / "metrics.json").read_text())["churn"]
        best = metrics["metrics"][metrics["best_model"]]
        a, b, c3, d = st.columns(4)
        a.metric("Model", metrics["best_model"])
        b.metric("Out-of-time ROC-AUC", f"{best['roc_auc']:.3f}")
        c3.metric("Top-decile lift", f"{best['top_decile_lift']:.2f}x")
        d.metric("Active customers scored", f"{len(scores):,}")
        st.caption("Population: customers who bought in the last 180 days. Churn = no purchase in the next 90 days. "
                   "Validated on a held-out *future* snapshot (no leakage).")
        band = st.multiselect("Risk band", ["High", "Medium", "Low"], default=["High", "Medium"])
        country = st.multiselect("Country ", sorted(scores["country"].unique()))
        view = scores[scores["risk_band"].isin(band)]
        if country:
            view = view[view["country"].isin(country)]
        a, b = st.columns([2, 3])
        a.plotly_chart(px.histogram(scores, x="churn_probability", nbins=30, title="Churn probability distribution",
                                    color_discrete_sequence=[NAVY]), width="stretch")
        b.metric("Revenue at risk (90 days) in selection", f"${view['value_at_risk_90d'].sum():,.0f}")
        b.dataframe(view.head(50), hide_index=True, width="stretch")
        st.download_button("Download win-back list", view.to_csv(index=False), "winback_list.csv", "text/csv")
        st.image(str(config.FIGURES_DIR / "04_churn_model.png"))
    except FileNotFoundError:
        st.info("Run `python main.py` to train the churn model.")

# ----------------------------------------------------------------------------------------- forecast
with tabs[5]:
    try:
        hist = load_table("revenue_history.csv")
        fc = load_table("revenue_forecast.csv")
        bt = load_table("forecast_backtest.csv")
        fig = go.Figure()
        fig.add_scatter(x=hist.iloc[:, 0], y=hist["net_revenue"], name="Actual", line=dict(color=NAVY, width=3))
        fig.add_scatter(x=fc["month"], y=fc["upper_80"], line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=fc["month"], y=fc["lower_80"], fill="tonexty", fillcolor="rgba(224,123,57,0.2)",
                        line=dict(width=0), name="80% interval", hoverinfo="skip")
        fig.add_scatter(x=fc["month"], y=fc["forecast"], name="Forecast", line=dict(color=ORANGE, width=3),
                        mode="lines+markers")
        fig.update_layout(title="Net revenue forecast", height=420, legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")
        st.subheader(f"Model selection (hold-out backtest): winner = {bt.loc[0, 'model']}")
        st.dataframe(bt.style.format({"mape": "{:.1%}", "rmse": "${:,.0f}", "bias": "{:+.1%}"}),
                     hide_index=True, width="stretch")
    except FileNotFoundError:
        st.info("Run `python main.py` to generate the forecast.")

# ---------------------------------------------------------------------------------------- marketing
with tabs[6]:
    try:
        ch = load_table("channels.csv")
        a, b = st.columns(2)
        a.plotly_chart(px.bar(ch, x="channel", y="cac", title="Customer acquisition cost (USD)",
                              color_discrete_sequence=[ORANGE]), width="stretch")
        fig = px.bar(ch, x="channel", y="ltv_to_cac", title="Realised LTV / CAC", color_discrete_sequence=["#2E8B57"])
        fig.add_hline(y=3, line_dash="dash", line_color="#C0392B", annotation_text="3x benchmark")
        b.plotly_chart(fig, width="stretch")
        st.dataframe(ch, hide_index=True, width="stretch")
    except FileNotFoundError:
        st.info("Run `python main.py` first.")
