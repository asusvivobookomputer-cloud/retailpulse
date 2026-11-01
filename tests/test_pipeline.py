import sqlite3

import numpy as np
import pandas as pd
import pytest

from retailpulse import analysis, etl, forecast, generate_data, ml, pipeline


# ------------------------------------------------------------------------------ data + ETL
def test_generator_writes_all_tables_with_dirty_rows(raw_dir):
    for name in etl.REQUIRED_COLUMNS:
        assert (raw_dir / f"{name}.csv").exists()
    customers = pd.read_csv(raw_dir / "customers.csv")
    assert customers["customer_id"].duplicated().any(), "raw data should contain duplicates to clean"


def test_etl_cleans_and_enforces_integrity(db, raw_dir):
    conn = sqlite3.connect(db)
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert conn.execute("SELECT COUNT(*) FROM order_items WHERE quantity <= 0").fetchone()[0] == 0
    ids = pd.read_sql_query("SELECT customer_id FROM customers", conn)["customer_id"]
    assert ids.is_unique
    countries = pd.read_sql_query("SELECT DISTINCT country FROM customers", conn)["country"]
    assert all(c == c.strip() and c[0].isupper() for c in countries)
    conn.close()


def test_etl_quality_log_records_actions(raw_dir, tmp_path):
    log = etl.run_etl(raw_dir, tmp_path / "x.db")
    assert {"table", "issue", "rows_affected", "action"} <= set(log.columns)
    assert (log["rows_affected"] > 0).all() and len(log) >= 5


def test_etl_fails_fast_on_missing_column(tmp_path, raw_dir):
    for f in raw_dir.glob("*.csv"):
        (tmp_path / f.name).write_bytes(f.read_bytes())
    pd.read_csv(tmp_path / "orders.csv").drop(columns=["status"]).to_csv(tmp_path / "orders.csv", index=False)
    with pytest.raises(ValueError, match="status"):
        etl.load_raw(tmp_path)


# --------------------------------------------------------------------------------- analytics
def test_sql_and_pandas_revenue_agree(db, lines):
    conn = analysis.connect(db)
    sql_total = conn.execute("SELECT SUM(net_revenue) FROM vw_order_lines WHERE status='completed'").fetchone()[0]
    conn.close()
    assert sql_total == pytest.approx(lines["net_revenue"].sum())


def test_kpis_are_internally_consistent(lines):
    k = analysis.kpis(lines)
    assert k["aov"] == pytest.approx(k["net_revenue"] / k["orders"])
    assert 0 < k["gross_margin_pct"] < 1
    assert 0 <= k["return_rate_pct"] < 0.5
    assert k["net_profit"] < k["net_revenue"]


def test_returned_lines_have_zero_net_revenue(lines):
    assert (lines.loc[lines["is_returned"] == 1, "net_revenue"] == 0).all()
    assert (lines.loc[lines["is_returned"] == 0, "net_revenue"] > 0).all()


def test_rfm_scores_and_segments(lines):
    r = analysis.rfm(lines)
    for col in "RFM":
        assert r[col].between(1, 5).all()
    assert r["customer_id"].is_unique
    assert not r["segment"].isna().any()
    s = analysis.rfm_segment_summary(r)
    assert s["customers"].sum() == len(r)
    assert s["revenue_share_pct"].sum() == pytest.approx(1.0, abs=1e-3)


def test_cohort_month_zero_is_100_percent_and_future_is_nan(lines):
    c = analysis.cohort_retention(lines)
    assert (c[0] == 1.0).all()
    assert np.isnan(c.iloc[-1, -1])                    # newest cohort cannot have month-12 data
    assert np.nanmax(c.drop(columns="cohort_size").values) <= 1.0


def test_pareto_is_sorted_and_classified(lines):
    p = analysis.pareto_products(lines)
    assert p["revenue"].is_monotonic_decreasing
    assert p["cum_revenue_share"].iloc[-1] == pytest.approx(1.0)
    assert set(p["abc_class"]) <= {"A", "B", "C"} and p["abc_class"].iloc[0] == "A"


def test_market_basket_recovers_planted_bundles(lines):
    b = analysis.market_basket(lines, min_pair_orders=5)
    assert len(b) > 0 and (b["lift"] > 1).all()


def test_named_sql_queries_run(db):
    conn = analysis.connect(db)
    for q in ["monthly_growth", "category_share", "top_customers_by_country", "return_rate_by_category"]:
        assert len(analysis.run_named_query(conn, q)) > 0
    with pytest.raises(KeyError):
        analysis.run_named_query(conn, "nope")
    conn.close()


# ---------------------------------------------------------------------------------------- ML
def test_churn_features_use_no_future_information(lines, customers):
    as_of = lines["order_date"].max() - pd.Timedelta(days=200)
    full = ml.build_features(lines, customers, as_of)
    truncated = ml.build_features(lines[lines["order_date"] <= as_of], customers, as_of)
    pd.testing.assert_frame_equal(full.sort_index(), truncated.sort_index())


def test_churn_label_matches_definition(lines, customers):
    as_of = lines["order_date"].max() - pd.Timedelta(days=120)
    X, y = ml.build_dataset(lines, customers, as_of)
    window = lines[(lines["order_date"] > as_of) & (lines["order_date"] <= as_of + pd.Timedelta(days=90))]
    buyers = set(window["customer_id"])
    assert all((cid not in buyers) == bool(v) for cid, v in y.items())
    assert (X["recency_days"] <= 180).all()


def test_churn_model_beats_random_out_of_time(lines, customers):
    res = ml.train_churn_model(lines, customers)
    assert res["metrics"][res["best_model"]]["roc_auc"] > 0.6
    scores = ml.score_customers(res["model"], lines, customers)
    assert scores["churn_probability"].between(0, 1).all()
    assert scores["value_at_risk_90d"].is_monotonic_decreasing


# ----------------------------------------------------------------------------------- forecast
def test_forecast_output_shape_and_sanity(lines):
    res = forecast.run_forecast(lines, horizon=6, test_months=3)
    f = res["forecast"]
    assert len(f) == 6 and (f["forecast"] > 0).all()
    assert (f["lower_80"] < f["forecast"]).all() and (f["forecast"] < f["upper_80"]).all()
    assert res["backtest"]["mape"].is_monotonic_increasing
    assert f["month"].iloc[0] == res["history"].index[-1] + pd.offsets.MonthBegin()


def test_partial_last_month_is_dropped(lines):
    cut = lines[lines["order_date"] <= lines["order_date"].max() - pd.Timedelta(days=10)]
    s = forecast.monthly_series(cut)
    assert s.index[-1] < cut["order_date"].max().to_period("M").to_timestamp()


# ----------------------------------------------------------------------------------- end-to-end
def test_full_pipeline_creates_every_artifact(raw_dir, tmp_path):
    out = tmp_path / "out"
    res = pipeline.run(generate=False, raw_dir=raw_dir, db_path=tmp_path / "e2e.db", out_dir=out)
    assert (out / "retail_report.xlsx").exists() and (out / "metrics.json").exists()
    assert len(list((out / "figures").glob("*.png"))) == 6
    assert (out / "models" / "churn_model.joblib").exists()
    assert res["ml"]["best_model"] in {"Logistic Regression", "Gradient Boosting"}
