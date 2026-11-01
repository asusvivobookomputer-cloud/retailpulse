# RetailPulse — End-to-End Retail Analytics & Machine Learning Pipeline

A production-style analytics project that takes raw relational CSV exports and turns them into
**clean data, a SQL warehouse, business insights, a churn-prediction model, a revenue forecast,
an executive Excel report and an interactive dashboard** — in one command.

```
raw CSVs ─▶ ETL & validation ─▶ SQLite (schema, FKs, views) ─▶ analytics ─┬▶ churn ML model
 (6 tables)   (data-quality log)   (window-function SQL)      RFM·cohorts  ├▶ revenue forecast
                                                              Pareto·basket├▶ Excel report (formulas + charts)
                                                              channel ROI  ├▶ PNG charts
                                                                           └▶ Streamlit dashboard
```

## Headline results (on the bundled dataset: 6,000 customers, 23k orders, 40k order lines)

| Area | Result |
|---|---|
| Net revenue / profit | **$1.77M / $0.67M** (37.8 % margin, 6.1 % return rate) |
| Customer value concentration | 16 % of customers ("Champions") generate **38 % of revenue** |
| Product Pareto | **57 of 120 products** make 80 % of revenue |
| Churn model (active customers) | **ROC-AUC 0.71** and **1.46× top-decile lift**, validated on a *future* snapshot |
| Revenue forecast | **4.2 % MAPE** on a 6-month hold-out (best of 3 models) |
| Acquisition channels | Organic LTV/CAC ≈ 30×, Paid Search ≈ 4.8× → budget re-allocation signal |

> The churn AUC is deliberately *honest*: the model is evaluated out-of-time, on active customers only,
> with no leakage (there is a unit test proving it). Inflated 0.95+ scores usually mean leakage.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py                 # generates data, runs the full pipeline (~10 s)
streamlit run app.py           # interactive dashboard
pytest                         # 18 automated tests
```

Outputs land in `outputs/`: `retail_report.xlsx`, `figures/*.png`, `tables/*.csv`, `metrics.json`,
`models/churn_model.joblib`.

## What is inside

| Module | Purpose |
|---|---|
| `retailpulse/generate_data.py` | Reproducible synthetic data with seasonality, churn, bundles, returns **and deliberately dirty records** |
| `retailpulse/etl.py` | Schema validation, cleaning rules, FK checks, audit log of every fix, SQLite load |
| `retailpulse/sql/` | `schema.sql` (PK/FK/CHECK), `views.sql` (analysis view), `analysis_queries.sql` (window functions, CTEs) |
| `retailpulse/analysis.py` | KPIs, monthly trend, RFM segmentation, cohort retention, ABC/Pareto, market-basket (support/confidence/lift), CAC·LTV·ROI by channel |
| `retailpulse/ml.py` | Churn model: multi-snapshot dataset, out-of-time validation, model comparison, permutation importance, win-back scoring |
| `retailpulse/forecast.py` | Seasonal-naive vs Ridge trend+season vs damped Holt-Winters, hold-out backtest, 80 % intervals |
| `retailpulse/report.py` | 9-sheet Excel report: live formulas, conditional formatting, native charts |
| `retailpulse/visuals.py` | Six publication-quality PNG charts |
| `app.py` | Streamlit dashboard with filters, 7 tabs and CSV downloads |
| `tests/` | 18 tests: ETL integrity, SQL↔pandas reconciliation, leakage, labels, forecast sanity, end-to-end |

## Use it on a client's data

1. Put CSVs with the same file names/columns in `data/raw/` (see `REQUIRED_COLUMNS` in `etl.py`).
2. Run `python main.py --skip-generate`.

Different columns? Adjust `REQUIRED_COLUMNS`, `sql/schema.sql` and `sql/views.sql` — the rest of the
pipeline works off the `vw_order_lines` view, so most changes stay in those three places.

## Key design decisions

* **Net revenue** = line revenue after order discount, excluding cancelled orders and returned items
  (returned goods are assumed restocked, so their COGS is also excluded). Defined once, in `views.sql`.
* **Churn** = an *active* customer (bought in the last 180 days) makes no purchase in the next 90 days.
  Features at snapshot `t` use only orders ≤ `t`; the newest snapshot is the hold-out set.
* **Model choice by evidence**: Logistic Regression beat Gradient Boosting out-of-time, so it ships.
  The pipeline picks the winner automatically.
* **Forecast intervals** come from backtest errors, not from a distributional assumption.
* **Everything is auditable**: the ETL log, SQL view and Excel formulas let a reviewer trace every number.

## Skills demonstrated

Python (pandas, NumPy, scikit-learn, matplotlib, Plotly) · SQL (DDL, constraints, views, window functions, CTEs) ·
data cleaning & validation · customer analytics (RFM, cohorts, LTV/CAC) · supervised ML with leakage-free validation ·
time-series forecasting · Excel automation (openpyxl) · dashboards (Streamlit) · testing (pytest) · reproducible project structure.

## Limitations

The bundled data is synthetic (so the project is shareable and reproducible). Metrics will differ on real data;
the value of the project is the method, the structure and the engineering discipline.
