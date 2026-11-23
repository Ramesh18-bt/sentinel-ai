"""SQLite persistence layer for SentinelAI.

The module owns the *single* source of truth for the physical schema. Every
other module (seed generator, quality engine, forensic tools, agent, API and
UI) reads and writes through these helpers so the column names never drift.

Design notes
------------
* ``sentinel.db`` runs in WAL mode so the FastAPI backend can write while the
  Streamlit UI and the read-only forensic tools keep reading concurrently.
* Connections are short lived and always committed/rolled back explicitly.
* The transform SQL lives on disk (``sql/transform_daily_customers.sql``) so the
  closed-loop "approve & apply fix" step can rewrite real files, not mocks.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv("SENTINEL_DB_PATH") or (PROJECT_ROOT / "sentinel.db")).resolve()
SQL_DIR = PROJECT_ROOT / "sql"
TRANSFORM_SQL_PATH = SQL_DIR / "transform_daily_customers.sql"
JOB_NAME = "transform_daily_customers"

# ---------------------------------------------------------------------------
# Transform SQL payloads (kept byte-identical to what lives on disk)
# ---------------------------------------------------------------------------

HEALTHY_TRANSFORM_SQL = """-- SentinelAI :: transform_daily_customers (healthy baseline)
-- Materialises the daily customer dimension for a single :run_day partition.
-- Guests without a registration_date keep the checkout timestamp as the
-- effective registration date, so no customer is silently dropped.

DELETE FROM dim_customers WHERE run_day = :run_day;

INSERT INTO dim_customers (user_id, run_day, email, effective_reg_date)
SELECT user_id, run_day, email, COALESCE(registration_date, created_at)
FROM raw_users
WHERE run_day = :run_day;
"""

BUGGY_TRANSFORM_SQL = """-- SentinelAI :: transform_daily_customers
-- Commit a8f92c1 (alex.dev) - "Fix downstream timestamp parse warning"
-- The added predicate removes guest checkouts from the customer dimension.

DELETE FROM dim_customers WHERE run_day = :run_day;

INSERT INTO dim_customers (user_id, run_day, email, effective_reg_date)
SELECT user_id, run_day, email, COALESCE(registration_date, created_at)
FROM raw_users
WHERE run_day = :run_day
  AND registration_date IS NOT NULL;
"""

# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS raw_users (
        user_id TEXT,
        run_day INTEGER,
        email TEXT,
        registration_date TEXT,
        created_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS dim_customers (
        user_id TEXT,
        run_day INTEGER,
        email TEXT,
        effective_reg_date TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pipeline_runs (
        run_id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_name TEXT,
        run_day INTEGER,
        status TEXT,
        rows_in INTEGER,
        rows_out INTEGER,
        duration_sec REAL,
        executed_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS metric_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        table_name TEXT,
        run_day INTEGER,
        row_count INTEGER,
        null_rate REAL,
        duplicate_count INTEGER,
        recorded_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lineage_edges (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        upstream_table TEXT,
        downstream_table TEXT,
        job_name TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS code_commits (
        commit_hash TEXT PRIMARY KEY,
        author TEXT,
        message TEXT,
        file_path TEXT,
        diff_text TEXT,
        committed_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS incidents (
        incident_id INTEGER PRIMARY KEY AUTOINCREMENT,
        table_name TEXT,
        run_day INTEGER,
        severity TEXT,
        anomaly_type TEXT,
        summary TEXT,
        status TEXT,
        investigation_json TEXT,
        proposed_sql TEXT,
        rollback_plan TEXT,
        approved_by TEXT,
        resolved_at TEXT,
        created_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT,
        content TEXT,
        tool_citations_json TEXT,
        created_at TEXT
    )
    """,
)

INDEX_STATEMENTS: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_raw_users_day ON raw_users (run_day)",
    "CREATE INDEX IF NOT EXISTS idx_dim_customers_day ON dim_customers (run_day)",
    "CREATE INDEX IF NOT EXISTS idx_metric_history_table_day ON metric_history (table_name, run_day)",
    "CREATE INDEX IF NOT EXISTS idx_pipeline_runs_job_day ON pipeline_runs (job_name, run_day)",
    "CREATE INDEX IF NOT EXISTS idx_incidents_day ON incidents (run_day, table_name)",
)

TABLE_NAMES: tuple[str, ...] = (
    "raw_users",
    "dim_customers",
    "pipeline_runs",
    "metric_history",
    "lineage_edges",
    "code_commits",
    "incidents",
    "chat_messages",
)

NULL_COLUMNS: dict[str, str] = {
    "raw_users": "registration_date",
    "stg_users_cleaned": "registration_date",
    "dim_customers": "effective_reg_date",
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def utc_now() -> str:
    """Timestamp helper shared by every writer (ISO-8601, UTC, seconds)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def get_connection(
    db_path: Path | str | None = None,
    *,
    bulk: bool = False,
) -> sqlite3.Connection:
    """Open a tuned SQLite connection (WAL, row factory, 30s busy timeout).

    ``bulk=True`` trades crash durability for throughput during the 1.4M-row
    baseline load (a 256MB page cache and ``synchronous=OFF``). The durable
    defaults are per-connection, so every normal request stays on
    ``synchronous=NORMAL`` inside WAL mode.
    """
    target = Path(db_path) if db_path else DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    if bulk:
        conn.execute("PRAGMA synchronous=OFF;")
        conn.execute("PRAGMA cache_size=-262144;")
    else:
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA cache_size=-64000;")
    return conn


@contextmanager
def connection(
    db_path: Path | str | None = None,
    *,
    bulk: bool = False,
) -> Iterator[sqlite3.Connection]:
    """Transactional connection context manager used across the platform."""
    conn = get_connection(db_path, bulk=bulk)
    try:
        conn.execute("BEGIN;")
        yield conn
        conn.execute("COMMIT;")
    except Exception:
        try:
            conn.execute("ROLLBACK;")
        except sqlite3.Error:
            pass
        raise
    finally:
        conn.close()


def init_schema(conn: sqlite3.Connection | None = None) -> None:
    """Create every table/index defined by the schema contract."""
    owned = conn is None
    conn = conn or get_connection()
    try:
        for statement in SCHEMA_STATEMENTS:
            conn.execute(statement)
        for statement in INDEX_STATEMENTS:
            conn.execute(statement)
    finally:
        if owned:
            conn.close()


def query_all(
    sql: str,
    params: Sequence[Any] | dict[str, Any] | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    """Run a read query and return a list of plain dicts."""
    owned = conn is None
    conn = conn or get_connection()
    try:
        cursor = conn.execute(sql, tuple(params) if isinstance(params, (list, tuple)) else (params or {}))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        if owned:
            conn.close()


def query_one(
    sql: str,
    params: Sequence[Any] | dict[str, Any] | None = None,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any] | None:
    """Run a read query and return the first row as a dict, else ``None``."""
    rows = query_all(sql, params=params, conn=conn)
    return rows[0] if rows else None


def execute(
    sql: str,
    params: Sequence[Any] | dict[str, Any] | None = None,
    conn: sqlite3.Connection | None = None,
) -> int:
    """Execute a write statement and return its ``lastrowid``."""
    owned = conn is None
    conn = conn or get_connection()
    try:
        cursor = conn.execute(sql, tuple(params) if isinstance(params, (list, tuple)) else (params or {}))
        if owned:
            conn.commit()
        return int(cursor.lastrowid or 0)
    finally:
        if owned:
            conn.close()


def execute_many(
    sql: str,
    rows: Sequence[Sequence[Any]],
    conn: sqlite3.Connection | None = None,
) -> int:
    """Batch insert helper; returns the number of submitted rows."""
    if not rows:
        return 0
    owned = conn is None
    conn = conn or get_connection()
    try:
        conn.executemany(sql, rows)
        if owned:
            conn.commit()
        return len(rows)
    finally:
        if owned:
            conn.close()


# ---------------------------------------------------------------------------
# Reset / transform helpers
# ---------------------------------------------------------------------------


def clear_all_tables(conn: sqlite3.Connection) -> None:
    """Truncate every table (including chat history) and reset sequences."""
    for table in TABLE_NAMES:
        conn.execute(f"DELETE FROM {table};")
    conn.execute("DELETE FROM sqlite_sequence;")


def read_transform_sql() -> str:
    """Return the current on-disk transform SQL (or the healthy default)."""
    if TRANSFORM_SQL_PATH.exists():
        return TRANSFORM_SQL_PATH.read_text(encoding="utf-8")
    write_transform_sql(HEALTHY_TRANSFORM_SQL)
    return HEALTHY_TRANSFORM_SQL


def write_transform_sql(sql_text: str) -> Path:
    """Persist a transform SQL file (closed-loop fix application)."""
    SQL_DIR.mkdir(parents=True, exist_ok=True)
    TRANSFORM_SQL_PATH.write_text(sql_text, encoding="utf-8")
    return TRANSFORM_SQL_PATH


def split_sql_statements(sql_text: str) -> list[str]:
    """Split a script into executable statements (comments are preserved)."""
    statements: list[str] = []
    buffer: list[str] = []
    for line in sql_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("--") and not buffer:
            continue
        buffer.append(line)
        if stripped.endswith(";"):
            statement = "\n".join(buffer).strip()
            statement = statement[:-1].strip()
            if statement:
                statements.append(statement)
            buffer = []
    tail = "\n".join(buffer).strip()
    if tail:
        statements.append(tail)
    return statements


def execute_transform_sql(
    run_day: int,
    sql_text: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Run the transform for ``run_day`` and return the materialised metrics."""
    script = sql_text if sql_text is not None else read_transform_sql()
    owned = conn is None
    conn = conn or get_connection()
    try:
        statements = split_sql_statements(script)
        for statement in statements:
            conn.execute(statement, {"run_day": run_day})
        rows_out = conn.execute(
            "SELECT COUNT(*) AS c FROM dim_customers WHERE run_day = ?", (run_day,)
        ).fetchone()[0]
        if owned:
            conn.commit()
        return {
            "run_day": run_day,
            "statements": len(statements),
            "rows_out": int(rows_out),
        }
    finally:
        if owned:
            conn.close()


def table_row_counts(conn: sqlite3.Connection | None = None) -> dict[str, int]:
    """Small diagnostic used by the API health probe."""
    counts: dict[str, int] = {}
    for table in TABLE_NAMES:
        row = query_one(f"SELECT COUNT(*) AS c FROM {table}", conn=conn)
        counts[table] = int(row["c"]) if row else 0
    return counts


def database_exists() -> bool:
    """True when the SQLite file has been created on disk."""
    return DB_PATH.exists()
