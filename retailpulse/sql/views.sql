-- One denormalised, analysis-ready line-level view.
-- Business rules (documented assumptions):
--   * net_revenue = line revenue after order discount, 0 if the line was returned
--   * returned goods are assumed to be restocked, so their COGS is also 0
DROP VIEW IF EXISTS vw_order_lines;
CREATE VIEW vw_order_lines AS
SELECT
    oi.order_item_id,
    o.order_id,
    o.customer_id,
    o.order_date,
    strftime('%Y-%m', o.order_date)                         AS order_month,
    o.sales_channel,
    o.payment_method,
    o.status,
    o.discount_pct,
    p.product_id,
    p.product_name,
    p.category,
    c.country,
    c.acquisition_channel,
    oi.quantity,
    oi.unit_price,
    ROUND(oi.quantity * oi.unit_price * (1 - o.discount_pct), 2)               AS gross_revenue,
    CASE WHEN r.return_id IS NULL THEN 0 ELSE 1 END                            AS is_returned,
    CASE WHEN r.return_id IS NULL
         THEN ROUND(oi.quantity * oi.unit_price * (1 - o.discount_pct), 2) ELSE 0 END AS net_revenue,
    CASE WHEN r.return_id IS NULL
         THEN ROUND(oi.quantity * oi.unit_price * (1 - o.discount_pct)
                    - oi.quantity * oi.unit_cost, 2) ELSE 0 END                AS net_profit
FROM order_items oi
JOIN orders    o ON o.order_id   = oi.order_id
JOIN products  p ON p.product_id = oi.product_id
JOIN customers c ON c.customer_id = o.customer_id
LEFT JOIN returns r ON r.order_item_id = oi.order_item_id;
