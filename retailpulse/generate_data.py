"""Generate a realistic relational retail dataset (6 CSV files).

The data is synthetic so the project is fully reproducible, but it is built with
real-world behaviour: seasonality, customer churn, bundles bought together,
channel-dependent customer quality, product returns, and *deliberately dirty*
records (duplicates, casing issues, orphans, outliers) so the ETL step has
meaningful work to do.

Swap in your own CSVs: drop files with the same names/columns into data/raw/
and run `python main.py --skip-generate`.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)

# category -> (product nouns, base price, return probability, paired category)
CATALOG = {
    "Electronics": (["Headphones", "Smartwatch", "Power Bank", "Bluetooth Speaker", "Webcam"], 85, 0.07, "Sports"),
    "Home & Kitchen": (["Blender", "Coffee Maker", "Cookware Set", "Vacuum Cleaner", "Air Fryer"], 60, 0.04, "Grocery"),
    "Fashion": (["Jacket", "Sneakers", "Backpack", "Sunglasses", "Hoodie"], 55, 0.15, "Beauty"),
    "Beauty": (["Face Serum", "Perfume", "Hair Dryer", "Makeup Kit", "Skincare Set"], 30, 0.05, "Fashion"),
    "Sports": (["Yoga Mat", "Dumbbell Set", "Running Belt", "Water Bottle", "Fitness Tracker"], 35, 0.04, "Electronics"),
    "Books": (["Novel", "Cookbook", "Self-Help Book", "Textbook", "Comic Collection"], 18, 0.02, "Toys"),
    "Toys": (["Building Blocks", "Puzzle", "RC Car", "Board Game", "Plush Toy"], 25, 0.03, "Books"),
    "Grocery": (["Coffee Beans", "Olive Oil", "Protein Bars", "Green Tea", "Dried Fruit Mix"], 12, 0.01, "Home & Kitchen"),
}
ADJECTIVES = ["Classic", "Pro", "Eco", "Smart", "Compact", "Premium"]

COUNTRIES = ["United States", "United Kingdom", "Germany", "Turkey", "Azerbaijan", "UAE"]
COUNTRY_P = [0.28, 0.17, 0.15, 0.14, 0.14, 0.12]

# acquisition channel -> (share, lifetime multiplier, order-rate multiplier, base CAC in USD)
CHANNELS = {
    "Organic Search": (0.30, 1.0, 1.0, 4.0),
    "Paid Search": (0.22, 0.7, 0.9, 18.0),
    "Social Media": (0.22, 0.6, 0.8, 14.0),
    "Email": (0.11, 1.3, 1.2, 6.0),
    "Referral": (0.15, 1.4, 1.2, 9.0),
}

# month-of-year demand index (Nov/Dec peak)
SEASONALITY = np.array([0.70, 0.65, 0.75, 0.80, 0.80, 0.75, 0.70, 0.75, 0.85, 0.90, 1.00, 0.95])


def _build_products(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    pid = 1
    for category, (nouns, base, _, _) in CATALOG.items():
        for noun in nouns:
            for adj in rng.choice(ADJECTIVES, size=3, replace=False):
                price = round(float(base * rng.lognormal(0, 0.35)), 0) - 0.01
                price = max(price, 2.99)
                cost = round(price * rng.uniform(0.45, 0.75), 2)
                rows.append((f"P{pid:03d}", f"{adj} {noun}", category, price, cost))
                pid += 1
    return pd.DataFrame(rows, columns=["product_id", "product_name", "category", "unit_price", "unit_cost"])


def _build_customers(rng: np.random.Generator, n: int, total_days: int, start: pd.Timestamp):
    # sign-ups grow over time
    w = np.exp(np.linspace(0, 1.0, total_days))
    signup_day = rng.choice(total_days, size=n, p=w / w.sum())
    ch_names = list(CHANNELS)
    channel = rng.choice(ch_names, size=n, p=[CHANNELS[c][0] for c in ch_names])
    life_mult = np.array([CHANNELS[c][1] for c in channel])
    rate_mult = np.array([CHANNELS[c][2] for c in channel])

    customers = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(1, n + 1)],
            "signup_date": (start + pd.to_timedelta(signup_day, unit="D")).strftime("%Y-%m-%d"),
            "country": rng.choice(COUNTRIES, size=n, p=COUNTRY_P),
            "acquisition_channel": channel,
            "age": np.clip(rng.normal(35, 11, n), 18, 70).round().astype(int),
            "gender": rng.choice(["Female", "Male"], size=n, p=[0.53, 0.47]),
        }
    )
    lifetime = rng.exponential(420 * life_mult)          # days a customer stays active
    rate = rng.gamma(2.0, 0.007, size=n) * rate_mult      # orders per active day
    return customers, signup_day, lifetime, rate


def _simulate_orders(rng, customers, signup_day, lifetime, rate, total_days, start) -> pd.DataFrame:
    month_of_day = (start + pd.to_timedelta(np.arange(total_days), unit="D")).month.values
    keep_p = SEASONALITY / SEASONALITY.max()
    cust_idx, days = [], []
    for i in range(len(customers)):
        s = int(signup_day[i])
        e = min(total_days - 1, s + int(lifetime[i]))
        window = e - s
        first = min(total_days - 1, s + int(rng.integers(0, 4)))
        d = [first]
        n_extra = rng.poisson(rate[i] * window)
        if n_extra:
            cand = s + rng.integers(0, window + 1, n_extra)
            cand = cand[rng.random(n_extra) < keep_p[month_of_day[cand] - 1]]
            d.extend(cand.tolist())
        cust_idx.extend([i] * len(d))
        days.extend(d)

    orders = pd.DataFrame({"cidx": cust_idx, "day": days}).sort_values(["day", "cidx"], kind="stable")
    orders = orders.reset_index(drop=True)
    n = len(orders)
    orders["order_id"] = [f"ORD-{i:06d}" for i in range(1, n + 1)]
    orders["customer_id"] = customers["customer_id"].values[orders["cidx"].values]
    orders["order_date"] = (start + pd.to_timedelta(orders["day"], unit="D")).dt.strftime("%Y-%m-%d")
    month = (start + pd.to_timedelta(orders["day"], unit="D")).dt.month.values

    orders["sales_channel"] = rng.choice(["Web", "Mobile App", "Store"], size=n, p=[0.55, 0.35, 0.10])
    orders["payment_method"] = rng.choice(
        ["Card", "PayPal", "Cash on Delivery", "Bank Transfer"], size=n, p=[0.60, 0.15, 0.15, 0.10]
    )
    orders["status"] = rng.choice(["completed", "cancelled"], size=n, p=[0.96, 0.04])
    disc = rng.choice([0.0, 0.05, 0.10, 0.15, 0.20], size=n, p=[0.55, 0.15, 0.15, 0.10, 0.05])
    black_friday = (month == 11) & (rng.random(n) < 0.5)
    disc[black_friday] = rng.choice([0.15, 0.20, 0.25, 0.30], size=black_friday.sum())
    orders["discount_pct"] = disc
    return orders[["order_id", "customer_id", "order_date", "sales_channel", "payment_method", "status",
                   "discount_pct", "day"]]


def _simulate_items(rng, orders, products, start) -> pd.DataFrame:
    n_prod = len(products)
    ranks = np.arange(1, n_prod + 1)
    pop = rng.permutation(1 / ranks ** 0.8)
    pop /= pop.sum()

    # fixed "bundle" partner living in the paired category -> strong association rules
    cat_idx = products.groupby("category").indices
    partner = np.empty(n_prod, dtype=int)
    for i, cat in enumerate(products["category"]):
        partner[i] = rng.choice(cat_idx[CATALOG[cat][3]])

    n_items = np.clip(1 + rng.poisson(0.8, len(orders)), 1, 5)
    first = rng.choice(n_prod, size=len(orders), p=pop)
    rows = []
    for k in range(len(orders)):
        chosen = [first[k]]
        for _ in range(n_items[k] - 1):
            nxt = partner[chosen[0]] if rng.random() < 0.30 else rng.choice(n_prod, p=pop)
            if nxt not in chosen:
                chosen.append(int(nxt))
        rows.extend((k, p) for p in chosen)

    items = pd.DataFrame(rows, columns=["oidx", "pidx"])
    n = len(items)
    items["quantity"] = rng.choice([1, 2, 3], size=n, p=[0.75, 0.17, 0.08])
    years = (orders["day"].values[items["oidx"].values]) / 365.25
    infl = 1 + 0.03 * years
    items["unit_price"] = (products["unit_price"].values[items["pidx"].values] * infl).round(2)
    items["unit_cost"] = (products["unit_cost"].values[items["pidx"].values] * infl).round(2)
    items["order_id"] = orders["order_id"].values[items["oidx"].values]
    items["product_id"] = products["product_id"].values[items["pidx"].values]
    items.insert(0, "order_item_id", np.arange(1, n + 1))
    return items[["order_item_id", "order_id", "product_id", "quantity", "unit_price", "unit_cost"]]


def _simulate_returns(rng, items, orders, products, total_days, start) -> pd.DataFrame:
    ret_p = products["category"].map({c: v[2] for c, v in CATALOG.items()}).values
    prod_idx = products.reset_index().set_index("product_id")["index"]
    p = ret_p[prod_idx.reindex(items["product_id"]).values]
    o = orders.set_index("order_id")
    completed = (o["status"].reindex(items["order_id"]).values == "completed")
    day = o["day"].reindex(items["order_id"]).values
    flag = completed & (rng.random(len(items)) < p)
    ret = items.loc[flag, ["order_item_id"]].copy()
    ret_day = day[flag] + rng.integers(3, 31, flag.sum())
    ok = ret_day < total_days
    ret, ret_day = ret[ok], ret_day[ok]
    ret["return_date"] = (start + pd.to_timedelta(ret_day, unit="D")).strftime("%Y-%m-%d")
    ret["reason"] = rng.choice(["Wrong size", "Defective", "Not as described", "Changed mind", "Late delivery"],
                               size=len(ret), p=[0.30, 0.20, 0.20, 0.20, 0.10])
    ret.insert(0, "return_id", np.arange(1, len(ret) + 1))
    return ret.reset_index(drop=True)


def _simulate_marketing(rng, customers) -> pd.DataFrame:
    c = customers.copy()
    c["month"] = pd.to_datetime(c["signup_date"]).dt.strftime("%Y-%m")
    new = c.groupby(["month", "acquisition_channel"]).size().rename("new_customers").reset_index()
    cac = {k: v[3] for k, v in CHANNELS.items()}
    new["spend"] = (new["new_customers"] * new["acquisition_channel"].map(cac)
                    * rng.uniform(0.85, 1.15, len(new))
                    + np.where(new["acquisition_channel"].isin(["Paid Search", "Social Media"]), 50, 0)).round(2)
    return new.rename(columns={"acquisition_channel": "channel"})[["month", "channel", "spend"]]


def _inject_dirt(rng, customers, orders, items):
    """Corrupt copies of the tables the way real exports are corrupted."""
    c = customers.copy()
    n = len(c)
    # inconsistent casing / whitespace in country
    idx = rng.choice(n, int(n * 0.03), replace=False)
    c.loc[idx, "country"] = c.loc[idx, "country"].map(lambda s: rng.choice([s.upper(), s.lower(), f" {s} "]))
    # missing and impossible ages
    c.loc[rng.choice(n, int(n * 0.025), replace=False), "age"] = np.nan
    k = max(3, int(n * 0.003))
    c.loc[rng.choice(n, k, replace=False), "age"] = rng.choice([0, 150, 999], size=k)
    # duplicate customer rows
    c = pd.concat([c, c.sample(max(3, int(n * 0.005)), random_state=1)], ignore_index=True)

    o = orders.copy()
    o = pd.concat([o, o.sample(max(3, int(len(o) * 0.002)), random_state=2)], ignore_index=True)

    it = items.copy()
    bad_q = rng.choice(len(it), max(3, int(len(it) * 0.0015)), replace=False)
    it.loc[bad_q, "quantity"] = -it.loc[bad_q, "quantity"]
    orphans = it.sample(max(3, int(len(it) * 0.002)), random_state=3).copy()
    orphans["order_id"] = "ORD-999999"
    orphans["order_item_id"] = np.arange(it["order_item_id"].max() + 1, it["order_item_id"].max() + 1 + len(orphans))
    it = pd.concat([it, orphans], ignore_index=True)
    return c, o, it


def generate(out_dir: Path | None = None, n_customers: int | None = None, seed: int | None = None,
             start_date: str | None = None, end_date: str | None = None, dirty: bool = True) -> dict[str, Path]:
    """Create the 6 raw CSV files and return their paths."""
    out_dir = Path(out_dir or config.DATA_RAW)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_customers = n_customers or config.N_CUSTOMERS
    rng = np.random.default_rng(seed if seed is not None else config.SEED)
    start = pd.Timestamp(start_date or config.START_DATE)
    end = pd.Timestamp(end_date or config.END_DATE)
    total_days = (end - start).days + 1

    products = _build_products(rng)
    customers, signup_day, lifetime, rate = _build_customers(rng, n_customers, total_days, start)
    orders = _simulate_orders(rng, customers, signup_day, lifetime, rate, total_days, start)
    items = _simulate_items(rng, orders, products, start)
    returns = _simulate_returns(rng, items, orders, products, total_days, start)
    marketing = _simulate_marketing(rng, customers)

    if dirty:
        customers, orders_out, items = _inject_dirt(rng, customers, orders, items)
    else:
        orders_out = orders
    orders_out = orders_out.drop(columns=["day"])

    tables = {"customers": customers, "products": products, "orders": orders_out,
              "order_items": items, "returns": returns, "marketing_spend": marketing}
    paths = {}
    for name, df in tables.items():
        paths[name] = out_dir / f"{name}.csv"
        df.to_csv(paths[name], index=False)
        log.info("wrote %-16s %8d rows -> %s", name, len(df), paths[name].name)
    return paths
