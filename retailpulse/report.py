"""Executive Excel report: formatted sheets, live formulas, conditional formatting, native charts.

Derived metrics (AOV, margin, MoM growth, CAC, ratios) are written as Excel *formulas*, not
pasted numbers, so the workbook stays auditable and recalculates when someone edits the data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.formatting.rule import ColorScaleRule, DataBarRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from . import config

FONT = "Arial"
NAVY = "1F4E79"
HEADER_FILL = PatternFill("solid", fgColor=NAVY)
HEADER_FONT = Font(name=FONT, bold=True, color="FFFFFF", size=10)
BODY_FONT = Font(name=FONT, size=10)
TITLE_FONT = Font(name=FONT, bold=True, size=16, color=NAVY)
NOTE_FONT = Font(name=FONT, italic=True, size=9, color="666666")
THIN = Border(bottom=Side(style="thin", color="D9D9D9"))

USD, USD2, PCT, INT = '"$"#,##0', '"$"#,##0.00', "0.0%", "#,##0"


def _clean(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    if v is pd.NaT or (isinstance(v, float) and np.isnan(v)):
        return None
    return v


def _header(ws, row: int, headers: list[str], col: int = 1) -> None:
    for j, h in enumerate(headers):
        c = ws.cell(row=row, column=col + j, value=h)
        c.font, c.fill = HEADER_FONT, HEADER_FILL
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[row].height = 30


def _write_df(ws, df: pd.DataFrame, row: int, formats: dict[str, str] | None = None, col: int = 1) -> None:
    formats = formats or {}
    _header(ws, row, list(df.columns), col)
    for i, rec in enumerate(df.itertuples(index=False), start=row + 1):
        for j, (name, val) in enumerate(zip(df.columns, rec)):
            c = ws.cell(row=i, column=col + j, value=_clean(val))
            c.font, c.border = BODY_FONT, THIN
            if name in formats:
                c.number_format = formats[name]


def _widths(ws, widths: dict[str, float] | None = None, default: float = 14) -> None:
    for j in range(1, ws.max_column + 1):
        ws.column_dimensions[get_column_letter(j)].width = default
    for k, v in (widths or {}).items():
        ws.column_dimensions[k].width = v


def _title(ws, text: str, note: str | None = None) -> None:
    ws["A1"], ws["A1"].font = text, TITLE_FONT
    if note:
        ws["A2"], ws["A2"].font = note, NOTE_FONT
    ws.sheet_view.showGridLines = False


# ------------------------------------------------------------------------------------------ sheets
def _summary(wb, res: dict, ml_res: dict, fc: dict) -> None:
    ws = wb.active
    ws.title = "Summary"
    _title(ws, "RetailPulse - Executive Summary", "Net figures exclude cancelled orders and returned items.")
    k = res["kpis"]
    _header(ws, 4, ["Metric", "Value"])
    inputs = [("Net revenue", k["net_revenue"], USD), ("Net profit", k["net_profit"], USD),
              ("Orders", k["orders"], INT), ("Customers", k["customers"], INT)]
    for i, (name, val, fmt) in enumerate(inputs, start=5):
        ws.cell(i, 1, name).font = BODY_FONT
        c = ws.cell(i, 2, val)
        c.font, c.number_format = Font(name=FONT, size=10, color="0000FF"), fmt   # blue = hard-coded input
    derived = [("Average order value (AOV)", "=IF(B7=0,0,B5/B7)", USD2),
               ("Gross margin", "=IF(B5=0,0,B6/B5)", PCT),
               ("Orders per customer", "=IF(B8=0,0,B7/B8)", "0.00"),
               ("Revenue per customer", "=IF(B8=0,0,B5/B8)", USD2)]
    for i, (name, f, fmt) in enumerate(derived, start=9):
        ws.cell(i, 1, name).font = BODY_FONT
        c = ws.cell(i, 2, f)
        c.font, c.number_format = BODY_FONT, fmt
    for i, (name, val, fmt) in enumerate([("Return rate (value)", k["return_rate_pct"], PCT),
                                          ("Repeat-customer rate", k["repeat_customer_pct"], PCT)], start=13):
        ws.cell(i, 1, name).font = BODY_FONT
        c = ws.cell(i, 2, val)
        c.font, c.number_format = Font(name=FONT, size=10, color="0000FF"), fmt

    m = ml_res["metrics"][ml_res["best_model"]]
    _header(ws, 16, ["Model & forecast", "Result"])
    rows = [("Churn model (best)", ml_res["best_model"], None),
            ("Out-of-time ROC-AUC", m["roc_auc"], "0.000"),
            ("Top-decile lift", m["top_decile_lift"], '0.00"x"'),
            ("Forecast model (best)", fc["best_model"], None),
            ("Forecast backtest MAPE", float(fc["backtest"].loc[0, "mape"]), PCT),
            ("Forecast next 6 months", float(fc["forecast"]["forecast"].sum()), USD)]
    for i, (name, val, fmt) in enumerate(rows, start=17):
        ws.cell(i, 1, name).font = BODY_FONT
        c = ws.cell(i, 2, val)
        c.font = BODY_FONT
        if fmt:
            c.number_format = fmt
    ws["A24"], ws["A24"].font = "Legend: blue = values loaded from the pipeline, black = Excel formulas.", NOTE_FONT
    _widths(ws, {"A": 34, "B": 26})


def _monthly(wb, res: dict) -> None:
    ws = wb.create_sheet("Monthly")
    _title(ws, "Monthly performance", "MoM growth, margin and AOV are live formulas.")
    df = res["monthly"][["order_month", "revenue", "profit", "orders", "customers"]]
    _write_df(ws, df, 4, {"revenue": USD, "profit": USD, "orders": INT, "customers": INT})
    _header(ws, 4, ["MoM growth", "Margin", "AOV"], col=6)
    n = len(df)
    for r in range(5, 5 + n):
        ws.cell(r, 6, "" if r == 5 else f"=IF(B{r-1}=0,\"\",B{r}/B{r-1}-1)").number_format = PCT
        ws.cell(r, 7, f"=IF(B{r}=0,0,C{r}/B{r})").number_format = PCT
        ws.cell(r, 8, f"=IF(D{r}=0,0,B{r}/D{r})").number_format = USD2
        for c in (6, 7, 8):
            ws.cell(r, c).font, ws.cell(r, c).border = BODY_FONT, THIN
    ws.freeze_panes = "A5"
    ch = LineChart()
    ch.title, ch.height, ch.width = "Net revenue & profit by month", 8.5, 20
    ch.add_data(Reference(ws, min_col=2, max_col=3, min_row=4, max_row=4 + n), titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=1, min_row=5, max_row=4 + n))
    ch.y_axis.numFmt = '"$"#,##0'
    ws.add_chart(ch, "J4")
    _widths(ws, {"A": 12})


def _rfm(wb, res: dict) -> None:
    ws = wb.create_sheet("RFM Segments")
    _title(ws, "RFM customer segments", "Revenue share is a formula over the revenue column.")
    s = res["rfm_summary"][["segment", "customers", "revenue", "avg_recency_days", "avg_frequency", "avg_monetary"]]
    _write_df(ws, s, 4, {"customers": INT, "revenue": USD, "avg_recency_days": "0", "avg_frequency": "0.0",
                         "avg_monetary": USD})
    _header(ws, 4, ["Revenue share"], col=7)
    n, last = len(s), 4 + len(s)
    for r in range(5, 5 + n):
        c = ws.cell(r, 7, f"=C{r}/SUM($C$5:$C${last})")
        c.number_format, c.font, c.border = PCT, BODY_FONT, THIN
    ws.conditional_formatting.add(f"G5:G{last}", DataBarRule(start_type="num", start_value=0, end_type="num",
                                                              end_value=0.5, color="5B9BD5"))
    ch = BarChart()
    ch.type, ch.title, ch.height, ch.width = "bar", "Revenue by RFM segment", 8.5, 16
    ch.add_data(Reference(ws, min_col=3, min_row=4, max_row=last), titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=1, min_row=5, max_row=last))
    ch.legend = None
    ws.add_chart(ch, "I4")
    _widths(ws, {"A": 22})


def _cohorts(wb, res: dict) -> None:
    ws = wb.create_sheet("Cohorts")
    _title(ws, "Monthly cohort retention", "Share of each acquisition cohort that purchased N months after its first order.")
    df = res["cohorts"].reset_index()
    fmts = {c: "0%" for c in df.columns if c not in ("cohort", "cohort_size")}
    df.columns = [f"M{c}" if isinstance(c, (int, np.integer)) else c for c in df.columns]
    fmts = {f"M{k}": v for k, v in fmts.items()}
    _write_df(ws, df, 4, {**fmts, "cohort_size": INT})
    last_col = get_column_letter(df.shape[1])
    ws.conditional_formatting.add(f"D5:{last_col}{4 + len(df)}",
                                  ColorScaleRule(start_type="num", start_value=0, start_color="FFFFFF",
                                                 end_type="num", end_value=0.35, end_color="2F75B5"))
    ws.freeze_panes = "C5"
    _widths(ws, default=8)
    ws.column_dimensions["A"].width = 11
    ws.column_dimensions["B"].width = 11


def _products(wb, res: dict) -> None:
    ws = wb.create_sheet("Products")
    _title(ws, "Product Pareto (ABC)", "A = first 80 % of revenue, B = next 15 %, C = last 5 %.")
    p = res["pareto"][["rank", "product_name", "category", "revenue", "profit", "units", "cum_revenue_share",
                       "abc_class"]]
    _write_df(ws, p, 4, {"revenue": USD, "profit": USD, "units": INT, "cum_revenue_share": PCT})
    _header(ws, 4, ["Margin"], col=9)
    for r in range(5, 5 + len(p)):
        c = ws.cell(r, 9, f"=IF(D{r}=0,0,E{r}/D{r})")
        c.number_format, c.font, c.border = PCT, BODY_FONT, THIN
    ws.freeze_panes = "A5"
    ws.auto_filter.ref = f"A4:I{4 + len(p)}"
    _widths(ws, {"B": 26, "C": 16})


def _marketing(wb, res: dict) -> None:
    ws = wb.create_sheet("Marketing")
    _title(ws, "Acquisition channel economics", "CAC, LTV/CAC and ROI are formulas. LTV = realised profit per customer to date.")
    d = res["channels"][["channel", "spend", "customers", "revenue", "profit"]]
    _write_df(ws, d, 4, {"spend": USD, "customers": INT, "revenue": USD, "profit": USD})
    _header(ws, 4, ["CAC", "LTV (profit)", "LTV / CAC", "ROI"], col=6)
    for r in range(5, 5 + len(d)):
        for col, f, fmt in [(6, f"=B{r}/C{r}", USD2), (7, f"=E{r}/C{r}", USD2),
                            (8, f"=G{r}/F{r}", '0.0"x"'), (9, f"=(E{r}-B{r})/B{r}", "0%")]:
            c = ws.cell(r, col, f)
            c.number_format, c.font, c.border = fmt, BODY_FONT, THIN
    ch = BarChart()
    ch.title, ch.height, ch.width = "LTV / CAC by channel", 8, 14
    ch.add_data(Reference(ws, min_col=8, min_row=4, max_row=4 + len(d)), titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=1, min_row=5, max_row=4 + len(d)))
    ch.legend = None
    ws.add_chart(ch, "A12")
    _widths(ws, {"A": 18})


def _churn(wb, scores: pd.DataFrame) -> None:
    ws = wb.create_sheet("Churn Risk")
    _title(ws, "Top 100 win-back targets", "Active customers ranked by expected revenue at risk over the next 90 days.")
    d = scores.head(100)[["customer_id", "country", "acquisition_channel", "recency_days", "frequency",
                          "monetary", "churn_probability", "risk_band", "value_at_risk_90d"]]
    _write_df(ws, d, 4, {"monetary": USD, "churn_probability": "0%", "value_at_risk_90d": USD2,
                         "frequency": INT, "recency_days": INT})
    ws.conditional_formatting.add(f"G5:G{4 + len(d)}",
                                  ColorScaleRule(start_type="num", start_value=0.3, start_color="FFFFFF",
                                                 end_type="num", end_value=0.9, end_color="F4A582"))
    ws.freeze_panes = "A5"
    ws.auto_filter.ref = f"A4:I{4 + len(d)}"
    _widths(ws, {"B": 16, "C": 18})


def _forecast(wb, fc: dict) -> None:
    ws = wb.create_sheet("Forecast")
    _title(ws, "Revenue forecast", f"Model: {fc['best_model']} (selected by hold-out backtest).")
    hist = fc["history"].reset_index()
    hist.columns = ["month", "actual"]
    hist["month"] = hist["month"].dt.strftime("%Y-%m")
    f = fc["forecast"].copy()
    f["month"] = f["month"].dt.strftime("%Y-%m")
    n = len(hist)
    _write_df(ws, hist, 4, {"actual": USD})
    _header(ws, 4, ["Forecast", "Lower 80%", "Upper 80%"], col=3)
    for i, rec in enumerate(f.itertuples(index=False), start=5 + n):
        ws.cell(i, 1, rec.month).font = BODY_FONT
        for j, v in enumerate([rec.forecast, rec.lower_80, rec.upper_80], start=3):
            c = ws.cell(i, j, float(v))
            c.number_format, c.font = USD, BODY_FONT
    total_row = 5 + n + len(f) + 1
    ws.cell(total_row, 1, "Next-6-month total").font = Font(name=FONT, bold=True, size=10)
    c = ws.cell(total_row, 3, f"=SUM(C{5 + n}:C{4 + n + len(f)})")
    c.number_format, c.font = USD, Font(name=FONT, bold=True, size=10)

    _header(ws, 4, ["Model", "MAPE", "RMSE", "Bias"], col=7)
    for i, rec in enumerate(fc["backtest"].itertuples(index=False), start=5):
        ws.cell(i, 7, rec.model).font = BODY_FONT
        for j, (v, fmt) in enumerate([(rec.mape, PCT), (rec.rmse, USD), (rec.bias, PCT)], start=8):
            c = ws.cell(i, j, float(v))
            c.number_format, c.font = fmt, BODY_FONT
    ch = LineChart()
    ch.title, ch.height, ch.width = "Actual vs forecast", 9, 20
    ch.add_data(Reference(ws, min_col=2, max_col=5, min_row=4, max_row=4 + n + len(f)), titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=1, min_row=5, max_row=4 + n + len(f)))
    ws.add_chart(ch, "G10")
    ws.freeze_panes = "A5"
    _widths(ws, {"A": 20, "G": 28})


def build_report(res: dict, ml_res: dict, scores: pd.DataFrame, fc: dict, quality: pd.DataFrame,
                 path: Path | None = None) -> Path:
    path = Path(path or config.OUTPUTS / "retail_report.xlsx")
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    _summary(wb, res, ml_res, fc)
    _monthly(wb, res)
    _rfm(wb, res)
    _cohorts(wb, res)
    _products(wb, res)
    _marketing(wb, res)
    _churn(wb, scores)
    _forecast(wb, fc)
    ws = wb.create_sheet("Data Quality")
    _title(ws, "ETL data-quality log", "Every cleaning action applied to the raw files.")
    _write_df(ws, quality, 4, {"rows_affected": INT})
    _widths(ws, {"A": 16, "B": 40, "C": 14, "D": 36})
    wb.save(path)
    return path
