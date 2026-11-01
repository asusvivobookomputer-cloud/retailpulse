"""Central configuration. Change values here, not inside the modules."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DB_PATH = DATA_PROCESSED / "retail.db"
OUTPUTS = ROOT / "outputs"
TABLES_DIR = OUTPUTS / "tables"
FIGURES_DIR = OUTPUTS / "figures"
MODELS_DIR = OUTPUTS / "models"
SQL_DIR = Path(__file__).resolve().parent / "sql"

# Synthetic data settings
SEED = 42
START_DATE = "2023-01-01"
END_DATE = "2026-09-30"
N_CUSTOMERS = 6000

# Analytics settings
CHURN_HORIZON_DAYS = 90        # a customer "churns" if no purchase in the next 90 days
ACTIVE_WINDOW_DAYS = 180       # only customers who bought in the last 180 days are 'active' and scored
CHURN_SNAPSHOTS = 5            # number of historical snapshots used to build the ML dataset
FORECAST_HORIZON = 6           # months
BACKTEST_MONTHS = 6            # hold-out length used to pick the forecasting model

RAW_TABLES = ["customers", "products", "orders", "order_items", "returns", "marketing_spend"]


def ensure_dirs() -> None:
    for p in (DATA_RAW, DATA_PROCESSED, TABLES_DIR, FIGURES_DIR, MODELS_DIR):
        p.mkdir(parents=True, exist_ok=True)
