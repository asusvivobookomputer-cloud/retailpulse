"""Monthly revenue forecasting with model selection by hold-out backtest.

Three transparent models are compared (no black boxes, no extra dependencies):
  1. Seasonal naive + growth  - same month last year x recent year-over-year growth
  2. Ridge trend + seasonality - regression on log revenue with a time trend and month dummies
  3. Damped Holt-Winters       - additive error/trend/season state-space smoothing on log revenue
The model with the lowest MAPE on the last `test_months` wins, is refit on all history and
produces the forecast; the prediction interval comes from the backtest residuals.
"""
from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from . import config

log = logging.getLogger(__name__)


def monthly_series(lines: pd.DataFrame) -> pd.Series:
    """Net revenue per month; a trailing partial month is dropped so it cannot bias the model."""
    s = lines.groupby(lines["order_date"].dt.to_period("M"))["net_revenue"].sum()
    last_day = lines["order_date"].max()
    if last_day != last_day.to_period("M").end_time.normalize():
        s = s.iloc[:-1]
    s.index = s.index.to_timestamp()
    return s.asfreq("MS")


# ---------------------------------------------------------------------------------------- models
def seasonal_naive(y: pd.Series, h: int) -> np.ndarray:
    vals = y.values
    growth = vals[-12:].sum() / vals[-24:-12].sum() if len(vals) >= 24 else 1.0
    growth = float(np.clip(growth, 0.7, 1.5))
    return np.array([vals[-12 + (k % 12)] * growth ** ((k // 12) + 1) for k in range(h)])


def _month_dummies(idx: pd.DatetimeIndex) -> np.ndarray:
    return np.eye(12)[idx.month.values - 1]


def ridge_trend_season(y: pd.Series, h: int, alpha: float = 1.0) -> np.ndarray:
    t = np.arange(len(y))
    X = np.column_stack([t / len(y), _month_dummies(y.index)])
    model = Ridge(alpha=alpha).fit(X, np.log(y.values))
    future_idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=h, freq="MS")
    Xf = np.column_stack([(np.arange(len(y), len(y) + h)) / len(y), _month_dummies(future_idx)])
    return np.exp(model.predict(Xf))


def _hw(y: np.ndarray, h: int, alpha: float, beta: float, gamma: float, phi: float = 0.97, m: int = 12):
    level = y[:m].mean()
    trend = (y[m:2 * m].mean() - y[:m].mean()) / m
    season = list(y[:m] - level)
    sse = 0.0
    for t in range(len(y)):
        s = season[t % m]
        pred = level + phi * trend + s
        if t >= m:
            sse += (y[t] - pred) ** 2
        prev = level
        level = alpha * (y[t] - s) + (1 - alpha) * (level + phi * trend)
        trend = beta * (level - prev) + (1 - beta) * phi * trend
        season[t % m] = gamma * (y[t] - level) + (1 - gamma) * s
    n = len(y)
    fc = [level + sum(phi ** i for i in range(1, k + 2)) * trend + season[(n + k) % m] for k in range(h)]
    return np.array(fc), sse


def holt_winters(y: pd.Series, h: int) -> np.ndarray:
    ly = np.log(y.values)
    grid = itertools.product([0.1, 0.3, 0.5], [0.05, 0.15], [0.1, 0.3, 0.5])
    best = min(grid, key=lambda p: _hw(ly, 1, *p)[1])
    fc, _ = _hw(ly, h, *best)
    return np.exp(fc)


MODELS = {"Seasonal naive + growth": seasonal_naive, "Ridge trend + seasonality": ridge_trend_season,
          "Damped Holt-Winters": holt_winters}


# ----------------------------------------------------------------------------------- evaluation
def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    return {"mape": float(np.mean(np.abs((actual - pred) / actual))),
            "rmse": float(np.sqrt(np.mean((actual - pred) ** 2))),
            "bias": float(np.mean(pred - actual) / np.mean(actual))}


def backtest(y: pd.Series, test_months: int = config.BACKTEST_MONTHS) -> tuple[pd.DataFrame, dict]:
    train, test = y.iloc[:-test_months], y.iloc[-test_months:]
    rows, preds = [], {}
    for name, fn in MODELS.items():
        p = fn(train, test_months)
        preds[name] = p
        rows.append({"model": name, **_metrics(test.values, p)})
    table = pd.DataFrame(rows).sort_values("mape").reset_index(drop=True)
    return table, preds


def run_forecast(lines: pd.DataFrame, horizon: int = config.FORECAST_HORIZON,
                 test_months: int = config.BACKTEST_MONTHS) -> dict:
    y = monthly_series(lines)
    table, bt_preds = backtest(y, test_months)
    best = table.loc[0, "model"]
    fc = MODELS[best](y, horizon)

    # interval from backtest log-errors of the winning model (80 % ~ +/-1.28 sigma)
    err = np.log(y.iloc[-test_months:].values) - np.log(bt_preds[best])
    sigma = max(float(np.sqrt(np.mean(err ** 2))), 0.03)
    idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=horizon, freq="MS")
    forecast = pd.DataFrame({"month": idx, "forecast": fc,
                             "lower_80": fc * np.exp(-1.28 * sigma), "upper_80": fc * np.exp(1.28 * sigma)})
    log.info("forecast: best model = %s (backtest MAPE %.1f%%)", best, 100 * table.loc[0, "mape"])
    return {"history": y, "forecast": forecast.round({"forecast": 2, "lower_80": 2, "upper_80": 2}), "backtest": table.round(4), "best_model": best,
            "backtest_pred": pd.DataFrame(bt_preds, index=y.index[-test_months:])}
