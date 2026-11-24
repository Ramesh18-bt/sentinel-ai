-- SentinelAI :: transform_daily_customers (healthy baseline)
-- Materialises the daily customer dimension for a single :run_day partition.
-- Guests without a registration_date keep the checkout timestamp as the
-- effective registration date, so no customer is silently dropped.

DELETE FROM dim_customers WHERE run_day = :run_day;

INSERT INTO dim_customers (user_id, run_day, email, effective_reg_date)
SELECT user_id, run_day, email, COALESCE(registration_date, created_at)
FROM raw_users
WHERE run_day = :run_day;
