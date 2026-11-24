"""Deterministic seed + failure simulator for SentinelAI.

Everything is generated set-based inside SQLite (recursive CTE + a coprime
modular hash for NULL placement) so 1.4M baseline rows land in well under two
seconds while remaining exactly reproducible - no NumPy needed.

Storyline
---------
* Days 1-14  : healthy baseline, 100k +/- small variance, 12% guest checkouts.
* Day 15 ok  : healthy run, 101,200 rows, quality engine passes.
* Day 15 bad : weekend promo pushes guest checkouts to 38.3% NULL, commit
  ``a8f92c1`` adds ``AND registration_date IS NOT NULL`` and the transform
  silently materialises 62,400 rows (-37.6%).
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

from core.database import (
    BUGGY_TRANSFORM_SQL,
    HEALTHY_TRANSFORM_SQL,
    JOB_NAME,
    clear_all_tables,
    connection,
    execute_transform_sql,
    init_schema,
    query_one,
    utc_now,
    write_transform_sql,
)
from core.quality_engine import evaluate_quality, record_metrics

BASELINE_DAYS = 14
TARGET_DAILY_ROWS = 100_000
BASELINE_NULL_PCT = 0.12
DAY15_RUN_DAY = 15
DAY15_HEALTHY_ROWS = 101_200
DAY15_ANOMALY_ROWS = 101_200
DAY15_ANOMALY_NULL_ROWS = 38_800  # 38.3% of 101,200 -> dim lands on exactly 62,400
ANOMALY_HASH = "a8f92c1"
ANOMALY_AUTHOR = "alex.dev"
ANOMALY_MESSAGE = "Fix downstream timestamp parse warning"
TRANSFORM_FILE = "sql/transform_daily_customers.sql"
EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)

# Offsets sum to exactly 0 so the 14-day mean is exactly 100,000 while the
# standard deviation stays realistically non-zero (z-score is meaningful).
BASELINE_OFFSETS: tuple[int, ...] = (
    1_800,
    -2_400,
    900,
    -1_500,
    2_100,
    -600,
    300,
    -1_200,
    1_500,
    -900,
    750,
    -1_350,
    1_200,
    -600,
)

HASH_MULTIPLIER = 7_919
GRID_A = 400
GRID_B = 300  # 120,000 generator slots > any daily partition


# ---------------------------------------------------------------------------
# Deterministic volumes
# ---------------------------------------------------------------------------


def baseline_volume(run_day: int) -> int:
    """Volume for a baseline day (days 1-14), exactly 100k on average."""
    return TARGET_DAILY_ROWS + BASELINE_OFFSETS[(run_day - 1) % len(BASELINE_OFFSETS)]


def _coprime_multiplier(n: int) -> int:
    """Return a multiplier coprime with ``n`` so NULL placement is exact."""
    candidate = HASH_MULTIPLIER
    while math.gcd(candidate, n) != 1:
        candidate += 2
    return candidate


def _day_timestamp(run_day: int) -> str:
    return (EPOCH + timedelta(days=run_day - 1, hours=2)).isoformat()


# ---------------------------------------------------------------------------
# Partition writers
# ---------------------------------------------------------------------------


def _insert_raw_partition(
    conn: sqlite3.Connection,
    run_day: int,
    row_count: int,
    null_rows: int,
) -> int:
    """Set-based insert of one ``raw_users`` partition (fast + exact NULLs).

    Rows come from a cross-joined recursive grid (``GRID_A * GRID_B`` slots)
    rather than a 100k-deep recursion, which is measurably faster in SQLite.
    The NULL placement uses a coprime modular hash so the guest-checkout count
    is *exactly* ``null_rows``, never approximate.
    """
    if row_count <= 0:
        return 0
    if row_count > GRID_A * GRID_B:
        raise ValueError(f"run_day {run_day}: row_count exceeds generator grid capacity")

    conn.execute("DELETE FROM raw_users WHERE run_day = ?", (run_day,))
    multiplier = _coprime_multiplier(row_count)
    conn.execute(
        """
        INSERT INTO raw_users (user_id, run_day, email, registration_date, created_at)
        WITH RECURSIVE
            g1(a) AS (SELECT 1 UNION ALL SELECT a + 1 FROM g1 WHERE a < :grid_a),
            g2(b) AS (SELECT 1 UNION ALL SELECT b + 1 FROM g2 WHERE b < :grid_b),
            seq AS (SELECT (g1.a - 1) * :grid_b + g2.b AS x FROM g1 CROSS JOIN g2)
        SELECT
            'U' || :run_day || '-' || seq.x AS user_id,
            :run_day AS run_day,
            'u' || seq.x || '.' || :run_day || '@sentinelmail.io' AS email,
            CASE
                WHEN ((seq.x * :mult) % :n) < :null_rows THEN NULL
                ELSE CASE (seq.x % 12)
                    WHEN 0 THEN '2023-01-15'
                    WHEN 1 THEN '2023-02-11'
                    WHEN 2 THEN '2023-03-08'
                    WHEN 3 THEN '2023-04-19'
                    WHEN 4 THEN '2023-05-06'
                    WHEN 5 THEN '2023-06-22'
                    WHEN 6 THEN '2023-07-03'
                    WHEN 7 THEN '2023-08-14'
                    WHEN 8 THEN '2023-09-27'
                    WHEN 9 THEN '2023-10-09'
                    WHEN 10 THEN '2023-11-18'
                    ELSE '2023-12-05'
                END
            END AS registration_date,
            :created_at AS created_at
        FROM seq
        WHERE seq.x <= :n;
        """,
        {
            "n": row_count,
            "run_day": run_day,
            "mult": multiplier,
            "null_rows": null_rows,
            "grid_a": GRID_A,
            "grid_b": GRID_B,
            "created_at": _day_timestamp(run_day),
        },
    )
    return row_count


def _log_pipeline_run(
    conn: sqlite3.Connection,
    run_day: int,
    status: str,
    rows_in: int,
    rows_out: int,
    duration_sec: float,
    executed_at: str | None = None,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO pipeline_runs
            (job_name, run_day, status, rows_in, rows_out, duration_sec, executed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            JOB_NAME,
            run_day,
            status,
            rows_in,
            rows_out,
            float(duration_sec),
            executed_at or _day_timestamp(run_day),
        ),
    )
    return int(cursor.lastrowid or 0)


def _seed_lineage(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM lineage_edges;")
    conn.executemany(
        """
        INSERT INTO lineage_edges (upstream_table, downstream_table, job_name)
        VALUES (?, ?, ?)
        """,
        [
            ("raw_users", "stg_users_cleaned", "ingest_users"),
            ("stg_users_cleaned", "dim_customers", JOB_NAME),
            ("dim_customers", "exec_revenue_dashboard", "build_exec_mart"),
        ],
    )


def _seed_clean_commits(conn: sqlite3.Connection) -> None:
    commits = [
        (
            "91b2e4a",
            "priya.analytics",
            "Add guest checkout timestamp fallback",
            TRANSFORM_FILE,
            "+ COALESCE(registration_date, created_at) AS effective_reg_date",
            (EPOCH + timedelta(days=2, hours=4)).isoformat(),
        ),
        (
            "4c81d0f",
            "marco.data",
            "Backfill dim_customers for days 1-7",
            TRANSFORM_FILE,
            "+ WHERE run_day = :run_day",
            (EPOCH + timedelta(days=8, hours=6)).isoformat(),
        ),
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO code_commits
            (commit_hash, author, message, file_path, diff_text, committed_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        commits,
    )


def _clear_day(conn: sqlite3.Connection, run_day: int) -> None:
    conn.execute("DELETE FROM raw_users WHERE run_day = ?", (run_day,))
    conn.execute("DELETE FROM dim_customers WHERE run_day = ?", (run_day,))
    conn.execute("DELETE FROM pipeline_runs WHERE run_day = ?", (run_day,))
    conn.execute("DELETE FROM metric_history WHERE run_day = ?", (run_day,))
    conn.execute("DELETE FROM incidents WHERE run_day = ?", (run_day,))


# ---------------------------------------------------------------------------
# Public simulation contract
# ---------------------------------------------------------------------------


def reset_and_seed_baseline() -> dict[str, Any]:
    """Wipe every table, restore healthy SQL and seed Days 1-14."""
    started = datetime.now(timezone.utc)
    with connection(bulk=True) as conn:
        init_schema(conn)
        clear_all_tables(conn)
        _seed_lineage(conn)
        _seed_clean_commits(conn)
        write_transform_sql(HEALTHY_TRANSFORM_SQL)

        total_raw = 0
        for run_day in range(1, BASELINE_DAYS + 1):
            row_count = baseline_volume(run_day)
            null_rows = int(round(row_count * BASELINE_NULL_PCT))
            _insert_raw_partition(conn, run_day, row_count, null_rows)
            result = execute_transform_sql(run_day, HEALTHY_TRANSFORM_SQL, conn=conn)
            # Baselines are generated with unique user_ids per partition, so the
            # duplicate scan (0.3s per partition) is skipped here; the quality
            # engine still computes duplicates authoritatively on every run.
            record_metrics("raw_users", run_day, None, conn, check_duplicates=False)
            record_metrics("dim_customers", run_day, None, conn, check_duplicates=False)
            _log_pipeline_run(
                conn,
                run_day=run_day,
                status="SUCCESS",
                rows_in=row_count,
                rows_out=int(result["rows_out"]),
                duration_sec=round(21.5 + (run_day % 5) * 0.9, 1),
            )
            total_raw += row_count

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    return {
        "action": "reset_and_seed_baseline",
        "status": "READY",
        "days_seeded": BASELINE_DAYS,
        "raw_users_rows": total_raw,
        "dim_customers_rows": query_one("SELECT COUNT(*) AS c FROM dim_customers")["c"],
        "elapsed_sec": round(elapsed, 2),
        "transform_sql_restored": True,
        "seeded_at": utc_now(),
    }


def simulate_day15_healthy() -> dict[str, Any]:
    """Land a clean Day 15: 101,200 rows in, 101,200 rows out, quality PASS."""
    with connection() as conn:
        init_schema(conn)
        _clear_day(conn, DAY15_RUN_DAY)
        conn.execute("DELETE FROM code_commits WHERE commit_hash = ?", (ANOMALY_HASH,))
        write_transform_sql(HEALTHY_TRANSFORM_SQL)

        null_rows = int(round(DAY15_HEALTHY_ROWS * BASELINE_NULL_PCT))
        _insert_raw_partition(conn, DAY15_RUN_DAY, DAY15_HEALTHY_ROWS, null_rows)
        result = execute_transform_sql(DAY15_RUN_DAY, HEALTHY_TRANSFORM_SQL, conn=conn)
        rows_out = int(result["rows_out"])
        _log_pipeline_run(
            conn,
            run_day=DAY15_RUN_DAY,
            status="SUCCESS",
            rows_in=DAY15_HEALTHY_ROWS,
            rows_out=rows_out,
            duration_sec=24.6,
            executed_at=_day_timestamp(DAY15_RUN_DAY),
        )
        verdict = evaluate_quality("dim_customers", DAY15_RUN_DAY, "raw_users", conn=conn)

    return {
        "action": "simulate_day15_healthy",
        "run_day": DAY15_RUN_DAY,
        "raw_users_rows": DAY15_HEALTHY_ROWS,
        "dim_customers_rows": rows_out,
        "null_registration_rows": null_rows,
        "quality_status": verdict["status"],
        "incident_id": verdict.get("incident_id"),
        "verdict": verdict,
        "completed_at": utc_now(),
    }


def inject_day15_anomaly() -> dict[str, Any]:
    """Inject the silent Day 15 failure and let the quality engine catch it."""
    with connection() as conn:
        init_schema(conn)
        _clear_day(conn, DAY15_RUN_DAY)

        _insert_raw_partition(
            conn, DAY15_RUN_DAY, DAY15_ANOMALY_ROWS, DAY15_ANOMALY_NULL_ROWS
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO code_commits
                (commit_hash, author, message, file_path, diff_text, committed_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                ANOMALY_HASH,
                ANOMALY_AUTHOR,
                ANOMALY_MESSAGE,
                TRANSFORM_FILE,
                "+ AND registration_date IS NOT NULL",
                _day_timestamp(DAY15_RUN_DAY),
            ),
        )
        write_transform_sql(BUGGY_TRANSFORM_SQL)
        result = execute_transform_sql(DAY15_RUN_DAY, BUGGY_TRANSFORM_SQL, conn=conn)
        rows_out = int(result["rows_out"])
        _log_pipeline_run(
            conn,
            run_day=DAY15_RUN_DAY,
            status="SUCCESS",
            rows_in=DAY15_ANOMALY_ROWS,
            rows_out=rows_out,
            duration_sec=26.1,
            executed_at=_day_timestamp(DAY15_RUN_DAY),
        )
        verdict = evaluate_quality("dim_customers", DAY15_RUN_DAY, "raw_users", conn=conn)

    baseline_mean = float(verdict["baseline"]["mean_row_count"]) or float(TARGET_DAILY_ROWS)
    return {
        "action": "inject_day15_anomaly",
        "run_day": DAY15_RUN_DAY,
        "raw_users_rows": DAY15_ANOMALY_ROWS,
        "dim_customers_rows": rows_out,
        "null_registration_rows": DAY15_ANOMALY_NULL_ROWS,
        "null_registration_pct": round(DAY15_ANOMALY_NULL_ROWS / DAY15_ANOMALY_ROWS * 100, 2),
        "drop_pct_vs_baseline": round((rows_out - baseline_mean) / baseline_mean * 100, 2),
        "pipeline_status": "SUCCESS",
        "silent_failure": True,
        "quality_status": verdict["status"],
        "incident_id": verdict.get("incident_id"),
        "verdict": verdict,
        "injected_at": utc_now(),
    }


def seed_demo() -> dict[str, Any]:
    """Convenience entrypoint: baseline + healthy Day 15 (clean demo state)."""
    baseline = reset_and_seed_baseline()
    healthy = simulate_day15_healthy()
    return {"baseline": baseline, "healthy_day15": healthy}


def _main() -> None:
    parser = argparse.ArgumentParser(description="SentinelAI deterministic seed simulator")
    parser.add_argument(
        "--mode",
        choices=("baseline", "healthy", "anomaly", "demo"),
        default="demo",
        help="which simulation step to execute (default: baseline + healthy day 15)",
    )
    args = parser.parse_args()

    if args.mode == "baseline":
        summary = reset_and_seed_baseline()
    elif args.mode == "healthy":
        summary = simulate_day15_healthy()
    elif args.mode == "anomaly":
        summary = inject_day15_anomaly()
    else:
        summary = seed_demo()

    import json

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    _main()
