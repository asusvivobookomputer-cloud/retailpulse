"""Customer churn prediction.

Population: *active* customers only (bought within the last 180 days) - this is the group a
retention team can actually act on. Including long-dead customers would inflate every metric.

Design (the part clients care about - no data leakage):
  * Several historical *snapshots* are taken. For each snapshot date `t` features are computed
    only from orders <= t, and the label says whether the customer did NOT buy in (t, t + 90 days].
  * Training uses older snapshots; the most recent snapshot is held out -> an honest
    *out-of-time* validation, the way the model would really be used.
  * Two models are compared (Logistic Regression baseline vs Gradient Boosting); the best by
    ROC-AUC is refit on all labelled snapshots and used to score customers today.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

log = logging.getLogger(__name__)

NUM_FEATURES = ["recency_days", "frequency", "monetary", "aov", "tenure_days", "avg_gap_days",
                "orders_per_month", "orders_last_90d", "orders_prev_90d", "recency_gap_ratio", "return_rate",
                "avg_discount", "n_categories", "age"]
CAT_FEATURES = ["country", "acquisition_channel", "gender"]


# ---------------------------------------------------------------------------------- features
def build_features(lines: pd.DataFrame, customers: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Customer-level features using ONLY information available on `as_of`."""
    hist = lines[lines["order_date"] <= as_of]
    orders = hist.groupby("order_id").agg(
        customer_id=("customer_id", "first"), order_date=("order_date", "first"),
        revenue=("net_revenue", "sum"), discount=("discount_pct", "first"),
    )
    g = orders.groupby("customer_id")
    f = pd.DataFrame({
        "last_order": g["order_date"].max(), "first_order": g["order_date"].min(),
        "frequency": g.size(), "monetary": g["revenue"].sum(), "avg_discount": g["discount"].mean(),
    })
    f["recency_days"] = (as_of - f["last_order"]).dt.days
    f["tenure_days"] = (as_of - f["first_order"]).dt.days
    f["aov"] = f["monetary"] / f["frequency"]
    f["orders_per_month"] = f["frequency"] / (f["tenure_days"] / 30.4 + 1)
    f["avg_gap_days"] = np.where(f["frequency"] > 1, f["tenure_days"] / (f["frequency"] - 1).clip(lower=1), np.nan)
    f["recency_gap_ratio"] = f["recency_days"] / f["avg_gap_days"]

    d90, d180 = as_of - pd.Timedelta(days=90), as_of - pd.Timedelta(days=180)
    f["orders_last_90d"] = orders[orders["order_date"] > d90].groupby("customer_id").size()
    f["orders_prev_90d"] = orders[(orders["order_date"] > d180) & (orders["order_date"] <= d90)] \
        .groupby("customer_id").size()
    f[["orders_last_90d", "orders_prev_90d"]] = f[["orders_last_90d", "orders_prev_90d"]].fillna(0)

    f["return_rate"] = hist.groupby("customer_id")["is_returned"].mean()
    f["n_categories"] = hist.groupby("customer_id")["category"].nunique()
    f = f.join(customers.set_index("customer_id")[["country", "acquisition_channel", "gender", "age"]])
    f["recency_gap_ratio"] = f["recency_gap_ratio"].replace([np.inf, -np.inf], np.nan)
    return f.drop(columns=["last_order", "first_order"])


def build_dataset(lines: pd.DataFrame, customers: pd.DataFrame, as_of: pd.Timestamp,
                  horizon: int = config.CHURN_HORIZON_DAYS) -> tuple[pd.DataFrame, pd.Series]:
    X = build_features(lines, customers, as_of)
    X = X[X["recency_days"] <= config.ACTIVE_WINDOW_DAYS]      # churn only makes sense for active customers
    window = lines[(lines["order_date"] > as_of) & (lines["order_date"] <= as_of + pd.Timedelta(days=horizon))]
    buyers = set(window["customer_id"])
    y = pd.Series(~X.index.isin(buyers), index=X.index, name="churned").astype(int)
    return X, y


def snapshot_dates(lines: pd.DataFrame, n: int = config.CHURN_SNAPSHOTS,
                   horizon: int = config.CHURN_HORIZON_DAYS) -> list[pd.Timestamp]:
    """Most recent snapshot first; each one has a fully observed label window."""
    last = lines["order_date"].max()
    return [last - pd.Timedelta(days=horizon * (i + 1)) for i in range(n)]


# ------------------------------------------------------------------------------------- models
def _preprocessor(scale: bool) -> ColumnTransformer:
    num = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])


def _candidates() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline([
            ("prep", _preprocessor(scale=True)),
            ("model", LogisticRegression(max_iter=1000, C=0.5)),
        ]),
        "Gradient Boosting": Pipeline([
            ("prep", _preprocessor(scale=False)),
            ("model", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=250,
                                                     l2_regularization=1.0, random_state=config.SEED)),
        ]),
    }


def _evaluate(y_true: pd.Series, prob: np.ndarray) -> dict:
    pred = (prob >= 0.5).astype(int)
    k = max(1, int(len(prob) * 0.10))
    top = np.argsort(-prob)[:k]
    base = float(y_true.mean())
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "precision@0.5": float(precision_score(y_true, pred, zero_division=0)),
        "recall@0.5": float(recall_score(y_true, pred, zero_division=0)),
        "top_decile_lift": float(y_true.values[top].mean() / base) if base else float("nan"),
        "base_churn_rate": base,
    }


def train_churn_model(lines: pd.DataFrame, customers: pd.DataFrame) -> dict:
    snaps = snapshot_dates(lines)
    test_snap, train_snaps = snaps[0], snaps[1:]
    parts = [build_dataset(lines, customers, s) for s in train_snaps]
    X_train = pd.concat([p[0] for p in parts])
    y_train = pd.concat([p[1] for p in parts])
    X_test, y_test = build_dataset(lines, customers, test_snap)
    log.info("churn dataset: train=%d rows (%d snapshots), test=%d rows (snapshot %s)",
             len(X_train), len(train_snaps), len(X_test), test_snap.date())

    metrics, fitted, curves = {}, {}, {}
    for name, pipe in _candidates().items():
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        metrics[name] = _evaluate(y_test, prob)
        fitted[name] = pipe
        fpr, tpr, _ = roc_curve(y_test, prob)
        curves[name] = (fpr, tpr)
        log.info("%-20s ROC-AUC=%.3f PR-AUC=%.3f top-decile lift=%.2f", name, metrics[name]["roc_auc"],
                 metrics[name]["pr_auc"], metrics[name]["top_decile_lift"])

    best = max(metrics, key=lambda k: metrics[k]["roc_auc"])
    imp = permutation_importance(fitted[best], X_test, y_test, scoring="roc_auc", n_repeats=5,
                                 random_state=config.SEED, n_jobs=1)
    importance = pd.DataFrame({"feature": X_test.columns, "importance": imp.importances_mean,
                               "std": imp.importances_std}).sort_values("importance", ascending=False)

    # final model: refit on every labelled snapshot (including the hold-out)
    final = _candidates()[best]
    final.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))
    return {"best_model": best, "model": final, "metrics": metrics, "importance": importance,
            "roc_curves": curves, "test_snapshot": str(test_snap.date()),
            "n_train": int(len(X_train)), "n_test": int(len(X_test))}


def score_customers(model: Pipeline, lines: pd.DataFrame, customers: pd.DataFrame,
                    as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Churn probability for every *active* customer as of today, plus a win-back priority score."""
    as_of = as_of or lines["order_date"].max()
    X = build_features(lines, customers, as_of)
    X = X[X["recency_days"] <= config.ACTIVE_WINDOW_DAYS]
    out = X[["recency_days", "frequency", "monetary", "aov", "country", "acquisition_channel"]].copy()
    out["churn_probability"] = model.predict_proba(X)[:, 1]
    out["risk_band"] = pd.cut(out["churn_probability"], [-0.01, 0.4, 0.7, 1.01], labels=["Low", "Medium", "High"])
    # expected revenue over the next 90 days that is "at stake" if the customer is lost
    rate_per_day = out["monetary"] / X["tenure_days"].clip(lower=90)
    out["value_at_risk_90d"] = (out["churn_probability"] * rate_per_day * config.CHURN_HORIZON_DAYS).round(2)
    return out.sort_values("value_at_risk_90d", ascending=False).reset_index().round(4)


def save_model(result: dict, path: Path | None = None) -> Path:
    path = Path(path or config.MODELS_DIR / "churn_model.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": result["model"], "best_model": result["best_model"],
                 "metrics": result["metrics"], "features": NUM_FEATURES + CAT_FEATURES}, path)
    return path
