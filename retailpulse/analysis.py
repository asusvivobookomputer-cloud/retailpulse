"""Business analytics on top of the SQLite database."""
from __future__ import annotations

import re
import sqlite3
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------------------- loading
def connect(db_path: Path | None = None) -> sqlite3.Connection:
    return sqlite3.connect(Path(db_path or config.DB_PATH))


def load_lines(conn: sqlite3.Connection) -> pd.DataFrame:
    """Completed order lines (one row per product on an order)."""
    return pd.read_sql_query(
        "SELECT * FROM vw_order_lines WHERE status = 'completed'", conn, parse_dates=["order_date"]
    )


def load_customers(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM customers", conn, parse_dates=["signup_date"])


def run_named_query(conn: sqlite3.Connection, name: str) -> pd.DataFrame:
    """Run one block of sql/analysis_queries.sql by its `-- name:` tag."""
    text = (config.SQL_DIR / "analysis_queries.sql").read_text()
    blocks = re.split(r"--\s*name:\s*(\w+)\s*\n", text)
    queries = dict(zip(blocks[1::2], blocks[2::2]))
    if name not in queries:
        raise KeyError(f"Unknown query '{name}'. Available: {list(queries)}")
    return pd.read_sql_query(queries[name].strip().rstrip(";"), conn)


# ------------------------------------------------------------------------------------------- KPIs
def kpis(lines: pd.DataFrame) -> dict:
    """Headline KPIs. Revenue/profit are net of returns."""
    orders = lines["order_id"].nunique()
    revenue = float(lines["net_revenue"].sum())
    profit = float(lines["net_profit"].sum())
    gross = float(lines["gross_revenue"].sum())
    per_cust = lines.groupby("customer_id")["order_id"].nunique()
    return {
        "net_revenue": revenue,
        "net_profit": profit,
        "orders": int(orders),
        "customers": int(lines["customer_id"].nunique()),
        "aov": revenue / orders if orders else 0.0,
        "gross_margin_pct": profit / revenue if revenue else 0.0,
        "return_rate_pct": 1 - revenue / gross if gross else 0.0,
        "repeat_customer_pct": float((per_cust >= 2).mean()) if len(per_cust) else 0.0,
        "orders_per_customer": float(per_cust.mean()) if len(per_cust) else 0.0,
    }


def monthly_sales(lines: pd.DataFrame) -> pd.DataFrame:
    m = lines.groupby("order_month").agg(
        revenue=("net_revenue", "sum"), profit=("net_profit", "sum"),
        orders=("order_id", "nunique"), customers=("customer_id", "nunique"),
    )
    m["aov"] = m["revenue"] / m["orders"]
    m["mom_growth"] = m["revenue"].pct_change()
    return m.round(4).reset_index()


def category_performance(lines: pd.DataFrame) -> pd.DataFrame:
    g = lines.groupby("category").agg(
        revenue=("net_revenue", "sum"), profit=("net_profit", "sum"),
        units=("quantity", "sum"), lines=("order_item_id", "count"), returned=("is_returned", "sum"),
    )
    g["margin_pct"] = g["profit"] / g["revenue"]
    g["return_rate_pct"] = g["returned"] / g["lines"]
    g["revenue_share_pct"] = g["revenue"] / g["revenue"].sum()
    return g.sort_values("revenue", ascending=False).reset_index()


# ------------------------------------------------------------------------------------------ RFM
def _rfm_segment(r: int, f: int) -> str:
    if r >= 4:
        return {5: "Champions", 4: "Champions", 3: "Loyal Customers", 2: "Potential Loyalists"}.get(f, "New Customers")
    if r == 3:
        return "Loyal Customers" if f >= 3 else "Promising"
    if f >= 4:
        return "Can't Lose Them"
    if f == 3:
        return "At Risk"
    return "About to Sleep" if r == 2 else "Hibernating"


def rfm(lines: pd.DataFrame, snapshot: pd.Timestamp | None = None) -> pd.DataFrame:
    """RFM scoring (1-5) + rule-based segmentation. R and M use quintiles, F uses fixed bins."""
    snapshot = snapshot or (lines["order_date"].max() + pd.Timedelta(days=1))
    c = lines.groupby("customer_id").agg(
        last_order=("order_date", "max"), frequency=("order_id", "nunique"), monetary=("net_revenue", "sum"),
    )
    c["recency"] = (snapshot - c["last_order"]).dt.days
    c["R"] = pd.qcut(c["recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    c["F"] = pd.cut(c["frequency"], [0, 1, 2, 3, 5, np.inf], labels=[1, 2, 3, 4, 5]).astype(int)
    c["M"] = pd.qcut(c["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    c["segment"] = [_rfm_segment(r, f) for r, f in zip(c["R"], c["F"])]
    return c.reset_index()


def rfm_segment_summary(rfm_df: pd.DataFrame) -> pd.DataFrame:
    s = rfm_df.groupby("segment").agg(
        customers=("customer_id", "count"), revenue=("monetary", "sum"),
        avg_recency_days=("recency", "mean"), avg_frequency=("frequency", "mean"), avg_monetary=("monetary", "mean"),
    )
    s["customer_share_pct"] = s["customers"] / s["customers"].sum()
    s["revenue_share_pct"] = s["revenue"] / s["revenue"].sum()
    return s.sort_values("revenue", ascending=False).round(4).reset_index()


# --------------------------------------------------------------------------------------- cohorts
def cohort_retention(lines: pd.DataFrame, max_age: int = 12) -> pd.DataFrame:
    """Monthly cohort retention matrix (share of a cohort that orders again N months later)."""
    t = lines[["customer_id", "order_date"]].copy()
    t["period"] = t["order_date"].dt.to_period("M")
    first = t.groupby("customer_id")["period"].transform("min")
    t["cohort"] = first
    t["age"] = (t["period"].dt.year - first.dt.year) * 12 + (t["period"].dt.month - first.dt.month)
    t = t[t["age"] <= max_age]
    counts = t.groupby(["cohort", "age"])["customer_id"].nunique().unstack(fill_value=0)
    counts = counts.reindex(columns=range(max_age + 1), fill_value=0)
    retention = counts.div(counts[0], axis=0)
    last = lines["order_date"].dt.to_period("M").max()
    for cohort in retention.index:                       # periods that have not happened yet -> NaN
        max_obs = (last.year - cohort.year) * 12 + (last.month - cohort.month)
        retention.loc[cohort, retention.columns > max_obs] = np.nan
    retention.insert(0, "cohort_size", counts[0])
    retention.index = retention.index.astype(str)
    retention.index.name = "cohort"
    return retention


# ---------------------------------------------------------------------------------- products
def pareto_products(lines: pd.DataFrame) -> pd.DataFrame:
    """ABC classification: A = products making the first 80 % of revenue, B = next 15 %, C = last 5 %."""
    p = lines.groupby(["product_id", "product_name", "category"]).agg(
        revenue=("net_revenue", "sum"), profit=("net_profit", "sum"), units=("quantity", "sum")
    ).sort_values("revenue", ascending=False).reset_index()
    p["cum_revenue_share"] = p["revenue"].cumsum() / p["revenue"].sum()
    prev = p["cum_revenue_share"].shift(fill_value=0)
    p["abc_class"] = np.where(prev < 0.80, "A", np.where(prev < 0.95, "B", "C"))
    p["rank"] = np.arange(1, len(p) + 1)
    return p


def market_basket(lines: pd.DataFrame, min_pair_orders: int = 25, top: int = 25) -> pd.DataFrame:
    """Pairwise association rules (support / confidence / lift) on multi-item orders."""
    baskets = lines.groupby("order_id")["product_name"].apply(lambda s: sorted(set(s)))
    n = len(baskets)
    item_cnt: Counter = Counter()
    pair_cnt: Counter = Counter()
    for items in baskets:
        item_cnt.update(items)
        if len(items) >= 2:
            pair_cnt.update(combinations(items, 2))
    rows = []
    for (a, b), c in pair_cnt.items():
        if c < min_pair_orders:
            continue
        support = c / n
        rows.append({
            "product_a": a, "product_b": b, "orders_together": c, "support": support,
            "confidence_a_to_b": c / item_cnt[a], "confidence_b_to_a": c / item_cnt[b],
            "lift": support / ((item_cnt[a] / n) * (item_cnt[b] / n)),
        })
    cols = ["product_a", "product_b", "orders_together", "support", "confidence_a_to_b",
            "confidence_b_to_a", "lift"]
    return pd.DataFrame(rows, columns=cols).sort_values("lift", ascending=False).head(top).round(4).reset_index(drop=True)


# ------------------------------------------------------------------------------------- marketing
def channel_economics(conn: sqlite3.Connection, lines: pd.DataFrame) -> pd.DataFrame:
    """CAC, realised LTV (profit per acquired customer to date), ROI and LTV/CAC by acquisition channel."""
    spend = pd.read_sql_query("SELECT channel, SUM(spend) AS spend FROM marketing_spend GROUP BY channel", conn)
    cust = pd.read_sql_query(
        "SELECT acquisition_channel AS channel, COUNT(*) AS customers FROM customers GROUP BY 1", conn)
    perf = lines.groupby("acquisition_channel").agg(
        revenue=("net_revenue", "sum"), profit=("net_profit", "sum"), orders=("order_id", "nunique")
    ).rename_axis("channel").reset_index()
    df = spend.merge(cust, on="channel").merge(perf, on="channel")
    df["cac"] = df["spend"] / df["customers"]
    df["ltv_profit"] = df["profit"] / df["customers"]
    df["ltv_to_cac"] = df["ltv_profit"] / df["cac"]
    df["roi"] = (df["profit"] - df["spend"]) / df["spend"]
    return df.sort_values("ltv_to_cac", ascending=False).round(4).reset_index(drop=True)


# ---------------------------------------------------------------------------------------- driver
def run_analysis(conn: sqlite3.Connection, out_dir: Path | None = None) -> dict:
    """Compute every analytical table, save as CSV, and return them in a dict."""
    out_dir = Path(out_dir or config.TABLES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = load_lines(conn)
    rfm_df = rfm(lines)
    results = {
        "kpis": kpis(lines),
        "monthly": monthly_sales(lines),
        "categories": category_performance(lines),
        "rfm": rfm_df,
        "rfm_summary": rfm_segment_summary(rfm_df),
        "cohorts": cohort_retention(lines),
        "pareto": pareto_products(lines),
        "basket": market_basket(lines),
        "channels": channel_economics(conn, lines),
    }
    for q in ["monthly_growth", "category_share", "top_customers_by_country", "return_rate_by_category"]:
        results[f"sql_{q}"] = run_named_query(conn, q)

    for name, obj in results.items():
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(out_dir / f"{name}.csv", index=name == "cohorts")
    pd.Series(results["kpis"]).to_json(out_dir / "kpis.json", indent=2)
    return results
