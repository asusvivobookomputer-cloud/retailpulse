-- Relational schema with primary keys, foreign keys and CHECK constraints.
PRAGMA foreign_keys = ON;

CREATE TABLE customers (
    customer_id         TEXT PRIMARY KEY,
    signup_date         TEXT NOT NULL,
    country             TEXT NOT NULL,
    acquisition_channel TEXT NOT NULL,
    age                 INTEGER CHECK (age IS NULL OR age BETWEEN 16 AND 100),
    gender              TEXT
);

CREATE TABLE products (
    product_id   TEXT PRIMARY KEY,
    product_name TEXT NOT NULL,
    category     TEXT NOT NULL,
    unit_price   REAL NOT NULL CHECK (unit_price > 0),
    unit_cost    REAL NOT NULL CHECK (unit_cost >= 0)
);

CREATE TABLE orders (
    order_id       TEXT PRIMARY KEY,
    customer_id    TEXT NOT NULL REFERENCES customers (customer_id),
    order_date     TEXT NOT NULL,
    sales_channel  TEXT,
    payment_method TEXT,
    status         TEXT NOT NULL CHECK (status IN ('completed', 'cancelled')),
    discount_pct   REAL NOT NULL DEFAULT 0 CHECK (discount_pct BETWEEN 0 AND 1)
);

CREATE TABLE order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id      TEXT    NOT NULL REFERENCES orders (order_id),
    product_id    TEXT    NOT NULL REFERENCES products (product_id),
    quantity      INTEGER NOT NULL CHECK (quantity > 0),
    unit_price    REAL    NOT NULL,
    unit_cost     REAL    NOT NULL
);

CREATE TABLE returns (
    return_id     INTEGER PRIMARY KEY,
    order_item_id INTEGER NOT NULL UNIQUE REFERENCES order_items (order_item_id),
    return_date   TEXT,
    reason        TEXT
);

CREATE TABLE marketing_spend (
    month   TEXT NOT NULL,
    channel TEXT NOT NULL,
    spend   REAL NOT NULL,
    PRIMARY KEY (month, channel)
);

CREATE INDEX idx_orders_customer ON orders (customer_id);
CREATE INDEX idx_orders_date     ON orders (order_date);
CREATE INDEX idx_items_order     ON order_items (order_id);
CREATE INDEX idx_items_product   ON order_items (product_id);
