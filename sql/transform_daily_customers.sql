-- SentinelAI :: transform_daily_customers
-- Commit a8f92c1 (alex.dev) - "Fix downstream timestamp parse warning"
-- The added predicate removes guest checkouts from the customer dimension.

DELETE FROM dim_customers WHERE run_day = :run_day;

INSERT INTO dim_customers (user_id, run_day, email, effective_reg_date)
SELECT user_id, run_day, email, COALESCE(registration_date, created_at)
FROM raw_users
WHERE run_day = :run_day
  AND registration_date IS NOT NULL;
