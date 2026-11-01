-- Showcase of advanced SQL (window functions, CTEs). Each block is loaded by name.

-- name: monthly_growth
WITH m AS (
    SELECT order_month, ROUND(SUM(net_revenue), 2) AS revenue, COUNT(DISTINCT order_id) AS orders
    FROM vw_order_lines WHERE status = 'completed' GROUP BY order_month
)
SELECT order_month, revenue, orders,
       ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY order_month))
             / LAG(revenue) OVER (ORDER BY order_month), 1)                                AS mom_growth_pct,
       ROUND(AVG(revenue) OVER (ORDER BY order_month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2) AS moving_avg_3m
FROM m ORDER BY order_month;

-- name: category_share
SELECT category,
       ROUND(SUM(net_revenue), 2)                                         AS revenue,
       ROUND(100.0 * SUM(net_revenue) / SUM(SUM(net_revenue)) OVER (), 1) AS revenue_share_pct,
       ROUND(100.0 * SUM(net_profit) / NULLIF(SUM(net_revenue), 0), 1)    AS margin_pct,
       RANK() OVER (ORDER BY SUM(net_revenue) DESC)                       AS revenue_rank
FROM vw_order_lines WHERE status = 'completed'
GROUP BY category ORDER BY revenue_rank;

-- name: top_customers_by_country
WITH cust AS (
    SELECT country, customer_id, ROUND(SUM(net_revenue), 2) AS revenue, COUNT(DISTINCT order_id) AS orders
    FROM vw_order_lines WHERE status = 'completed' GROUP BY country, customer_id
), ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY country ORDER BY revenue DESC) AS rn FROM cust
)
SELECT country, rn AS rank_in_country, customer_id, revenue, orders
FROM ranked WHERE rn <= 5 ORDER BY country, rn;

-- name: return_rate_by_category
SELECT category,
       COUNT(*)                                          AS lines_sold,
       SUM(is_returned)                                  AS lines_returned,
       ROUND(100.0 * SUM(is_returned) / COUNT(*), 2)     AS return_rate_pct
FROM vw_order_lines WHERE status = 'completed'
GROUP BY category ORDER BY return_rate_pct DESC;
