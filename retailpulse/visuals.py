"""Static, portfolio-ready charts (PNG) saved to outputs/figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

PALETTE = {"primary": "#1F4E79", "accent": "#E07B39", "good": "#2E8B57", "bad": "#C0392B", "grey": "#8C8C8C"}
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 13, "figure.dpi": 110})


def _save(fig, name: str, out_dir: Path) -> Path:
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def revenue_forecast(fc: dict, out_dir: Path) -> Path:
    hist, f = fc["history"], fc["forecast"]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(hist.index, hist.values, color=PALETTE["primary"], lw=2, label="Actual net revenue")
    ax.plot([hist.index[-1], f["month"].iloc[0]], [hist.values[-1], f["forecast"].iloc[0]],
            color=PALETTE["accent"], lw=1.2, ls=":")
    ax.plot(f["month"], f["forecast"], color=PALETTE["accent"], lw=2.2, marker="o", label=f"Forecast ({fc['best_model']})")
    ax.fill_between(f["month"], f["lower_80"], f["upper_80"], color=PALETTE["accent"], alpha=0.2, label="80% interval")
    ax.axvline(hist.index[-1], color=PALETTE["grey"], ls="--", lw=1)
    ax.set_title("Monthly net revenue and 6-month forecast")
    ax.yaxis.set_major_formatter(lambda v, _: f"${v/1000:,.0f}k")
    ax.legend(frameon=False, loc="upper left")
    return _save(fig, "01_revenue_forecast.png", out_dir)


def cohort_heatmap(cohorts: pd.DataFrame, out_dir: Path) -> Path:
    data = cohorts.drop(columns="cohort_size")
    data = data.iloc[::2]                                       # every 2nd cohort keeps labels readable
    fig, ax = plt.subplots(figsize=(11, 6.5))
    im = ax.imshow(data.values.astype(float), cmap="Blues", vmin=0, vmax=0.5, aspect="auto")
    ax.set_xticks(range(data.shape[1]), [str(c) for c in data.columns])
    ax.set_yticks(range(data.shape[0]), data.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.3 else "#222")
    ax.set_xlabel("Months since first purchase")
    ax.set_title("Cohort retention (share of cohort ordering in month N)")
    fig.colorbar(im, ax=ax, fraction=0.03, format=lambda v, _: f"{v:.0%}")
    return _save(fig, "02_cohort_retention.png", out_dir)


def rfm_segments(summary: pd.DataFrame, out_dir: Path) -> Path:
    s = summary.sort_values("revenue_share_pct")
    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(s))
    ax.barh(y - 0.2, s["customer_share_pct"], 0.4, color=PALETTE["grey"], label="% of customers")
    ax.barh(y + 0.2, s["revenue_share_pct"], 0.4, color=PALETTE["primary"], label="% of revenue")
    ax.set_yticks(y, s["segment"])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("RFM segments: customer share vs revenue share")
    ax.legend(frameon=False, loc="lower right")
    return _save(fig, "03_rfm_segments.png", out_dir)


def churn_model_report(ml_res: dict, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    for name, (fpr, tpr) in ml_res["roc_curves"].items():
        auc = ml_res["metrics"][name]["roc_auc"]
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC {auc:.2f})")
    ax.plot([0, 1], [0, 1], color=PALETTE["grey"], ls="--")
    ax.set(xlabel="False positive rate", ylabel="True positive rate")
    ax.set_title("Out-of-time ROC curve")
    ax.legend(frameon=False, loc="lower right")
    imp = ml_res["importance"].head(8).iloc[::-1]
    axes[1].barh(imp["feature"], imp["importance"], xerr=imp["std"], color=PALETTE["primary"])
    axes[1].set_title("Permutation importance (drop in AUC)")
    return _save(fig, "04_churn_model.png", out_dir)


def pareto_chart(pareto: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    colors = pareto["abc_class"].map({"A": PALETTE["primary"], "B": PALETTE["accent"], "C": PALETTE["grey"]})
    ax.bar(pareto["rank"], pareto["revenue"], color=colors, width=0.85)
    ax.set_ylabel("Net revenue")
    ax.yaxis.set_major_formatter(lambda v, _: f"${v/1000:,.0f}k")
    ax2 = ax.twinx()
    ax2.plot(pareto["rank"], pareto["cum_revenue_share"], color=PALETTE["bad"], lw=2)
    ax2.axhline(0.8, color=PALETTE["bad"], ls=":", lw=1)
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax2.spines["right"].set_visible(True)
    n_a = int((pareto["abc_class"] == "A").sum())
    ax.set_title(f"Product Pareto: {n_a} of {len(pareto)} products (class A) generate 80% of revenue")
    ax.set_xlabel("Products ranked by revenue")
    return _save(fig, "05_product_pareto.png", out_dir)


def channel_economics(ch: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    c = ch.sort_values("cac")
    axes[0].bar(c["channel"], c["cac"], color=PALETTE["accent"])
    axes[0].set_title("Customer acquisition cost (USD)")
    axes[1].bar(c["channel"], c["ltv_to_cac"], color=PALETTE["good"])
    axes[1].axhline(3, color=PALETTE["bad"], ls="--", lw=1)
    axes[1].set_title("Realised LTV / CAC (3x = healthy)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
    return _save(fig, "06_channel_economics.png", out_dir)


def make_all(results: dict, ml_res: dict, fc: dict, out_dir: Path | None = None) -> list[Path]:
    out_dir = Path(out_dir or config.FIGURES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    return [revenue_forecast(fc, out_dir), cohort_heatmap(results["cohorts"], out_dir),
            rfm_segments(results["rfm_summary"], out_dir), churn_model_report(ml_res, out_dir),
            pareto_chart(results["pareto"], out_dir), channel_economics(results["channels"], out_dir)]
