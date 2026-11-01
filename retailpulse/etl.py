"""ETL: raw CSV -> validated, cleaned tables -> SQLite database (with a data-quality log)."""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "customers": ["customer_id", "signup_date", "country", "acquisition_channel", "age", "gender"],
    "products": ["product_id", "product_name", "category", "unit_price", "unit_cost"],
    "orders": ["order_id", "customer_id", "order_date", "sales_channel", "payment_method", "status", "discount_pct"],
    "order_items": ["order_item_id", "order_id", "product_id", "quantity", "unit_price", "unit_cost"],
    "returns": ["return_id", "order_item_id", "return_date", "reason"],
    "marketing_spend": ["month", "channel", "spend"],
}


class QualityLog:
    """Collects every cleaning action so the pipeline is auditable."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, table: str, issue: str, affected: int, action: str) -> None:
        if affected:
            self.rows.append({"table": table, "issue": issue, "rows_affected": int(affected), "action": action})
            log.info("DQ %-16s %-38s %6d  -> %s", table, issue, affected, action)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["table", "issue", "rows_affected", "action"])


def load_raw(raw_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Read the raw CSV files and fail early if a file or column is missing."""
    raw_dir = Path(raw_dir or config.DATA_RAW)
    tables = {}
    for name, cols in REQUIRED_COLUMNS.items():
        path = raw_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")
        df = pd.read_csv(path)
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"{name}.csv is missing columns: {sorted(missing)}")
        tables[name] = df[cols].copy()
        log.info("loaded %-16s %8d rows", name, len(df))
    return tables


def clean(tables: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Apply validation / cleaning rules. Returns (clean_tables, quality_log)."""
    dq = QualityLog()
    t = {k: v.copy() for k, v in tables.items()}

    # ---- customers -------------------------------------------------------------------------
    c = t["customers"]
    before = c["country"].copy()
    c["country"] = c["country"].astype(str).str.strip().str.title()
    c["country"] = c["country"].replace({"Uae": "UAE", "Usa": "United States"})
    dq.add("customers", "inconsistent country casing/whitespace", (before != c["country"]).sum(), "standardised")
    c["signup_date"] = pd.to_datetime(c["signup_date"], errors="coerce")
    dq.add("customers", "unparseable signup_date", c["signup_date"].isna().sum(), "row dropped")
    c = c.dropna(subset=["signup_date"])
    dup = c.duplicated("customer_id", keep="first")
    dq.add("customers", "duplicate customer_id", dup.sum(), "kept first occurrence")
    c = c[~dup].copy()
    bad_age = c["age"].notna() & ~c["age"].between(16, 100)
    dq.add("customers", "impossible age (<16 or >100)", bad_age.sum(), "set to NULL")
    c.loc[bad_age, "age"] = np.nan
    dq.add("customers", "missing age", c["age"].isna().sum(), "kept as NULL (imputed in ML step)")
    c["signup_date"] = c["signup_date"].dt.strftime("%Y-%m-%d")
    t["customers"] = c

    # ---- products --------------------------------------------------------------------------
    p = t["products"]
    dup = p.duplicated("product_id")
    dq.add("products", "duplicate product_id", dup.sum(), "kept first occurrence")
    p = p[~dup]
    bad = (p["unit_price"] <= 0) | (p["unit_cost"] < 0)
    dq.add("products", "non-positive price / negative cost", bad.sum(), "row dropped")
    t["products"] = p[~bad]

    # ---- orders ----------------------------------------------------------------------------
    o = t["orders"]
    dup = o.duplicated("order_id")
    dq.add("orders", "duplicate order_id", dup.sum(), "kept first occurrence")
    o = o[~dup].copy()
    o["order_date"] = pd.to_datetime(o["order_date"], errors="coerce")
    dq.add("orders", "unparseable order_date", o["order_date"].isna().sum(), "row dropped")
    o = o.dropna(subset=["order_date"])
    orphan = ~o["customer_id"].isin(t["customers"]["customer_id"])
    dq.add("orders", "unknown customer_id (orphan)", orphan.sum(), "row dropped")
    o = o[~orphan].copy()
    o["discount_pct"] = o["discount_pct"].fillna(0).clip(0, 1)
    o["order_date"] = o["order_date"].dt.strftime("%Y-%m-%d")
    t["orders"] = o

    # ---- order_items -----------------------------------------------------------------------
    it = t["order_items"]
    dup = it.duplicated("order_item_id")
    dq.add("order_items", "duplicate order_item_id", dup.sum(), "kept first occurrence")
    it = it[~dup]
    bad_q = it["quantity"].isna() | (it["quantity"] <= 0)
    dq.add("order_items", "non-positive quantity", bad_q.sum(), "row dropped")
    it = it[~bad_q]
    orphan = ~it["order_id"].isin(t["orders"]["order_id"])
    dq.add("order_items", "unknown order_id (orphan)", orphan.sum(), "row dropped")
    it = it[~orphan]
    orphan = ~it["product_id"].isin(t["products"]["product_id"])
    dq.add("order_items", "unknown product_id (orphan)", orphan.sum(), "row dropped")
    t["order_items"] = it[~orphan].astype({"quantity": int})

    # ---- returns ---------------------------------------------------------------------------
    r = t["returns"]
    orphan = ~r["order_item_id"].isin(t["order_items"]["order_item_id"])
    dq.add("returns", "unknown order_item_id (orphan)", orphan.sum(), "row dropped")
    r = r[~orphan]
    dup = r.duplicated("order_item_id")
    dq.add("returns", "item returned twice", dup.sum(), "kept first occurrence")
    t["returns"] = r[~dup]

    # ---- marketing -------------------------------------------------------------------------
    m = t["marketing_spend"].drop_duplicates(["month", "channel"])
    t["marketing_spend"] = m[m["spend"].notna()]

    return t, dq.to_frame()


def build_database(tables: dict[str, pd.DataFrame], db_path: Path | None = None) -> Path:
    """Create the SQLite database (schema + data + indexes + views)."""
    db_path = Path(db_path or config.DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript((config.SQL_DIR / "schema.sql").read_text())
        for name in ["customers", "products", "orders", "order_items", "returns", "marketing_spend"]:
            tables[name].to_sql(name, conn, if_exists="append", index=False)
            log.info("db: inserted %-16s %8d rows", name, len(tables[name]))
        conn.executescript((config.SQL_DIR / "views.sql").read_text())
        conn.commit()
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError(f"Foreign-key violations found: {violations[:5]}")
    finally:
        conn.close()
    return db_path


def run_etl(raw_dir: Path | None = None, db_path: Path | None = None) -> pd.DataFrame:
    """Full ETL run. Returns the data-quality log."""
    tables = load_raw(raw_dir)
    cleaned, quality = clean(tables)
    build_database(cleaned, db_path)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    quality.to_csv(config.TABLES_DIR / "data_quality_log.csv", index=False)
    return quality
