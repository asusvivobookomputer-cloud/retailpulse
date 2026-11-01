"""Orchestrates the whole project: generate -> ETL -> analytics -> ML -> forecast -> visuals -> Excel."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

from . import analysis, config, etl, forecast, generate_data, ml, report, visuals

log = logging.getLogger(__name__)


def run(generate: bool = True, raw_dir: Path | None = None, db_path: Path | None = None,
        out_dir: Path | None = None) -> dict:
    t0 = time.time()
    config.ensure_dirs()
    raw_dir = Path(raw_dir or config.DATA_RAW)
    out_dir = Path(out_dir or config.OUTPUTS)
    tables_dir, figures_dir = out_dir / "tables", out_dir / "figures"
    for d in (tables_dir, figures_dir, out_dir / "models"):
        d.mkdir(parents=True, exist_ok=True)

    if generate:
        log.info("[1/7] generating synthetic raw data")
        generate_data.generate(raw_dir)
    else:
        log.info("[1/7] using existing raw data in %s", raw_dir)

    log.info("[2/7] ETL: validate, clean, load into SQLite")
    quality = etl.run_etl(raw_dir, db_path)
    quality.to_csv(tables_dir / "data_quality_log.csv", index=False)

    log.info("[3/7] analytics: KPIs, RFM, cohorts, Pareto, basket, channels")
    conn = analysis.connect(db_path)
    try:
        results = analysis.run_analysis(conn, tables_dir)
        lines = analysis.load_lines(conn)
        customers = analysis.load_customers(conn)
    finally:
        conn.close()

    log.info("[4/7] machine learning: churn model with out-of-time validation")
    ml_res = ml.train_churn_model(lines, customers)
    scores = ml.score_customers(ml_res["model"], lines, customers)
    scores.to_csv(tables_dir / "churn_scores.csv", index=False)
    ml_res["importance"].to_csv(tables_dir / "churn_feature_importance.csv", index=False)
    ml.save_model(ml_res, out_dir / "models" / "churn_model.joblib")

    log.info("[5/7] forecasting: 3 models, hold-out backtest, 6-month forecast")
    fc = forecast.run_forecast(lines)
    fc["forecast"].to_csv(tables_dir / "revenue_forecast.csv", index=False)
    fc["backtest"].to_csv(tables_dir / "forecast_backtest.csv", index=False)
    fc["history"].rename("net_revenue").to_csv(tables_dir / "revenue_history.csv")

    log.info("[6/7] charts")
    visuals.make_all(results, ml_res, fc, figures_dir)

    log.info("[7/7] Excel report")
    xlsx = report.build_report(results, ml_res, scores, fc, quality, out_dir / "retail_report.xlsx")

    metrics = {"kpis": results["kpis"], "churn": {"best_model": ml_res["best_model"], "metrics": ml_res["metrics"],
                                                  "test_snapshot": ml_res["test_snapshot"],
                                                  "n_train": ml_res["n_train"], "n_test": ml_res["n_test"]},
               "forecast": {"best_model": fc["best_model"], "backtest": fc["backtest"].to_dict("records")}}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    log.info("done in %.1fs -> %s", time.time() - t0, out_dir)
    return {"results": results, "ml": ml_res, "scores": scores, "forecast": fc, "quality": quality, "excel": xlsx}
