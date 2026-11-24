"""Statistical data-quality engine for SentinelAI.

The engine is deliberately dependency-free (stdlib statistics only) so a
suspicious metric can be evaluated inside a worker without pulling pandas in.
It compares the *current* run against a 14-day rolling baseline built from
``metric_history`` and raises a CRITICAL incident when any guardrail trips.
"""

from __future__ import annotations

import math
import sqlite3
import statistics
from typing import Any

from core.database import (
    JOB_NAME,
    NULL_COLUMNS,
    connection,
    query_all,
    query_one,
    utc_now,
)

WINDOW_DAYS = 14
ROW_COUNT_DROP_PCT = -15.0
Z_SCORE_THRESHOLD = -3.0
NULL_RATE_SPIKE_PP = 10.0
DEFAULT_TABLE = "dim_customers"
DEFAULT_UPSTREAM = "raw_users"


# ---------------------------------------------------------------------------
# Profiling primitives
# ---------------------------------------------------------------------------


def _null_column(table_name: str) -> str | None:
    return NULL_COLUMNS.get(table_name)


def compute_table_metrics(
    table_name: str,
    run_day: int,
    conn: sqlite3.Connection | None = None,
    check_duplicates: bool = True,
) -> dict[str, Any]:
    """Profile one partition in a single scan: rows, NULL rate, duplicate keys."""
    null_column = _null_column(table_name)
    if null_column:
        row = query_one(
            f"""
            SELECT COUNT(*) AS row_count,
                   COALESCE(SUM({null_column} IS NULL), 0) AS null_count
            FROM {table_name}
            WHERE run_day = ?
            """,
            (run_day,),
            conn=conn,
        )
    else:
        row = query_one(
            f"SELECT COUNT(*) AS row_count, 0 AS null_count FROM {table_name} WHERE run_day = ?",
            (run_day,),
            conn=conn,
        )
    row_count = int(row["row_count"]) if row else 0
    nulls = int(row["null_count"]) if row else 0
    null_rate = round((nulls / row_count) * 100.0, 4) if row_count else 0.0

    duplicate_count = 0
    if check_duplicates and "user_id" in _table_columns(table_name, conn):
        dup_row = query_one(
            f"SELECT COUNT(*) - COUNT(DISTINCT user_id) AS dup FROM {table_name} WHERE run_day = ?",
            (run_day,),
            conn=conn,
        )
        duplicate_count = max(int(dup_row["dup"] or 0), 0) if dup_row else 0

    return {
        "table_name": table_name,
        "run_day": run_day,
        "row_count": row_count,
        "null_rate": null_rate,
        "duplicate_count": duplicate_count,
    }


def _table_columns(table_name: str, conn: sqlite3.Connection | None = None) -> list[str]:
    rows = query_all(f"PRAGMA table_info({table_name})", conn=conn)
    return [str(row["name"]) for row in rows]


def record_metrics(
    table_name: str,
    run_day: int,
    metrics: dict[str, Any] | None = None,
    conn: sqlite3.Connection | None = None,
    check_duplicates: bool = True,
) -> dict[str, Any]:
    """Upsert a metric snapshot for ``(table_name, run_day)``."""
    metrics = metrics or compute_table_metrics(
        table_name, run_day, conn=conn, check_duplicates=check_duplicates
    )

    def _write(active: sqlite3.Connection) -> dict[str, Any]:
        active.execute(
            "DELETE FROM metric_history WHERE table_name = ? AND run_day = ?",
            (table_name, run_day),
        )
        active.execute(
            """
            INSERT INTO metric_history
                (table_name, run_day, row_count, null_rate, duplicate_count, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                table_name,
                run_day,
                int(metrics["row_count"]),
                float(metrics["null_rate"]),
                int(metrics["duplicate_count"]),
                utc_now(),
            ),
        )
        return metrics

    if conn is not None:
        return _write(conn)
    with connection() as owned:
        return _write(owned)


# ---------------------------------------------------------------------------
# Baseline statistics
# ---------------------------------------------------------------------------


def baseline_stats(
    table_name: str,
    run_day: int,
    window: int = WINDOW_DAYS,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """14-day rolling mean / std-dev of row_count, null_rate and duplicates."""
    rows = query_all(
        """
        SELECT run_day, row_count, null_rate, duplicate_count
        FROM metric_history
        WHERE table_name = ? AND run_day < ?
        ORDER BY run_day DESC
        LIMIT ?
        """,
        (table_name, run_day, window),
        conn=conn,
    )
    history = sorted(rows, key=lambda item: int(item["run_day"]))
    counts = [float(item["row_count"]) for item in history]
    nulls = [float(item["null_rate"]) for item in history]
    dups = [float(item["duplicate_count"]) for item in history]

    mean_rows = statistics.fmean(counts) if counts else 0.0
    std_rows = statistics.pstdev(counts) if len(counts) > 1 else 0.0
    mean_null = statistics.fmean(nulls) if nulls else 0.0
    std_null = statistics.pstdev(nulls) if len(nulls) > 1 else 0.0
    mean_dups = statistics.fmean(dups) if dups else 0.0

    return {
        "table_name": table_name,
        "window_days": len(history),
        "days": [int(item["run_day"]) for item in history],
        "mean_row_count": round(mean_rows, 2),
        "std_row_count": round(std_rows, 2),
        "mean_null_rate": round(mean_null, 4),
        "std_null_rate": round(std_null, 4),
        "mean_duplicate_count": round(mean_dups, 2),
        "min_row_count": int(min(counts)) if counts else 0,
        "max_row_count": int(max(counts)) if counts else 0,
    }


# ---------------------------------------------------------------------------
# Guardrails and incident creation
# ---------------------------------------------------------------------------


def _check(
    name: str,
    passed: bool,
    observed: Any,
    threshold: Any,
    detail: str,
    blocking: bool = True,
) -> dict[str, Any]:
    """A guardrail result. ``blocking`` checks can raise/resolve incidents.

    Non-blocking checks are *explanatory* signals (for example a business-driven
    NULL spike in the upstream feed). They escalate severity and appear in the
    verdict, but they never keep the served table in an incident state once the
    row count and key integrity are healthy again.
    """
    return {
        "check": name,
        "passed": bool(passed),
        "blocking": bool(blocking),
        "observed": observed,
        "threshold": threshold,
        "detail": detail,
    }


def _upsert_incident(
    conn: sqlite3.Connection,
    table_name: str,
    run_day: int,
    severity: str,
    anomaly_type: str,
    summary: str,
) -> int:
    existing = conn.execute(
        """
        SELECT incident_id FROM incidents
        WHERE table_name = ? AND run_day = ? AND status <> 'RESOLVED'
        ORDER BY incident_id DESC
        LIMIT 1
        """,
        (table_name, run_day),
    ).fetchone()
    if existing:
        incident_id = int(existing["incident_id"])
        conn.execute(
            """
            UPDATE incidents
            SET severity = ?, anomaly_type = ?, summary = ?, status = 'OPEN',
                investigation_json = NULL, proposed_sql = NULL, rollback_plan = NULL,
                approved_by = NULL, resolved_at = NULL, created_at = ?
            WHERE incident_id = ?
            """,
            (severity, anomaly_type, summary, utc_now(), incident_id),
        )
        return incident_id
    cursor = conn.execute(
        """
        INSERT INTO incidents
            (table_name, run_day, severity, anomaly_type, summary, status,
             investigation_json, proposed_sql, rollback_plan, approved_by,
             resolved_at, created_at)
        VALUES (?, ?, ?, ?, ?, 'OPEN', NULL, NULL, NULL, NULL, NULL, ?)
        """,
        (table_name, run_day, severity, anomaly_type, summary, utc_now()),
    )
    return int(cursor.lastrowid or 0)


def evaluate_quality(
    table_name: str = DEFAULT_TABLE,
    run_day: int = 15,
    upstream_table: str = DEFAULT_UPSTREAM,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any]:
    """Evaluate guardrails for a run and persist a CRITICAL incident if needed.

    Returns a verdict payload used verbatim by the API and the UI scorecard.
    """
    if conn is not None:
        return _evaluate(conn, table_name, run_day, upstream_table)
    with connection() as owned:
        init = owned
        _ensure_schema(init)
        return _evaluate(init, table_name, run_day, upstream_table)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    from core.database import init_schema

    init_schema(conn)


def _evaluate(
    conn: sqlite3.Connection,
    table_name: str,
    run_day: int,
    upstream_table: str,
) -> dict[str, Any]:
    metrics = record_metrics(table_name, run_day, conn=conn)
    baseline = baseline_stats(table_name, run_day, conn=conn)

    row_count = int(metrics["row_count"])
    mean_rows = float(baseline["mean_row_count"]) or 0.0
    std_rows = float(baseline["std_row_count"]) or 0.0

    if mean_rows > 0:
        pct_change = round(((row_count - mean_rows) / mean_rows) * 100.0, 2)
    else:
        pct_change = 0.0

    if std_rows > 0:
        z_score = round((row_count - mean_rows) / std_rows, 2)
    else:
        z_score = 0.0 if row_count >= mean_rows else -99.0

    checks: list[dict[str, Any]] = []
    drop_triggered = pct_change <= ROW_COUNT_DROP_PCT or z_score <= Z_SCORE_THRESHOLD
    checks.append(
        _check(
            "row_count_drop",
            not drop_triggered,
            {"row_count": row_count, "pct_change": pct_change, "z_score": z_score},
            {"pct_change_lte": ROW_COUNT_DROP_PCT, "z_score_lte": Z_SCORE_THRESHOLD},
            (
                f"{table_name} materialised {row_count:,} rows vs 14-day mean "
                f"{mean_rows:,.0f} ({pct_change:+.1f}%, z={z_score:+.2f})"
            ),
        )
    )

    own_null_metric = float(metrics["null_rate"])
    own_null_baseline = float(baseline["mean_null_rate"])
    own_null_delta = round(own_null_metric - own_null_baseline, 2)
    checks.append(
        _check(
            "null_rate_spike",
            own_null_delta <= NULL_RATE_SPIKE_PP,
            {"null_rate": own_null_metric, "baseline_null_rate": own_null_baseline, "delta_pp": own_null_delta},
            {"delta_pp_gt": NULL_RATE_SPIKE_PP},
            f"{table_name} NULL rate {own_null_metric:.2f}% vs baseline {own_null_baseline:.2f}%",
        )
    )

    checks.append(
        _check(
            "duplicate_user_ids",
            int(metrics["duplicate_count"]) == 0,
            {"duplicate_count": int(metrics["duplicate_count"])},
            {"duplicate_count_gt": 0},
            f"{table_name} duplicate user_id rows: {int(metrics['duplicate_count']):,}",
        )
    )

    upstream_metrics = record_metrics(upstream_table, run_day, conn=conn)
    upstream_baseline = baseline_stats(upstream_table, run_day, conn=conn)
    upstream_null_delta = round(
        float(upstream_metrics["null_rate"]) - float(upstream_baseline["mean_null_rate"]), 2
    )
    checks.append(
        _check(
            "upstream_null_rate_spike",
            upstream_null_delta <= NULL_RATE_SPIKE_PP,
            {
                "table_name": upstream_table,
                "null_rate": float(upstream_metrics["null_rate"]),
                "baseline_null_rate": float(upstream_baseline["mean_null_rate"]),
                "delta_pp": upstream_null_delta,
                "row_count": int(upstream_metrics["row_count"]),
            },
            {"delta_pp_gt": NULL_RATE_SPIKE_PP},
            (
                f"{upstream_table} registration nullification at "
                f"{float(upstream_metrics['null_rate']):.2f}% vs "
                f"{float(upstream_baseline['mean_null_rate']):.2f}% baseline"
            ),
            blocking=False,
        )
    )

    failed = [item for item in checks if not item["passed"] and item["blocking"]]
    informational = [item for item in checks if not item["passed"] and not item["blocking"]]
    anomaly_parts: list[str] = []
    if not checks[0]["passed"]:
        anomaly_parts.append("ROW_COUNT_DROP")
    if not checks[1]["passed"] or not checks[3]["passed"]:
        anomaly_parts.append("NULL_RATE_SPIKE")
    if not checks[2]["passed"]:
        anomaly_parts.append("DUPLICATE_KEYS")

    verdict: dict[str, Any] = {
        "status": "CRITICAL" if failed else "PASS",
        "severity": "CRITICAL" if failed else "INFO",
        "failed_checks": [item["check"] for item in failed],
        "informational_flags": [item["check"] for item in informational],
        "table_name": table_name,
        "run_day": run_day,
        "metrics": metrics,
        "baseline": baseline,
        "upstream": {
            "metrics": upstream_metrics,
            "baseline": upstream_baseline,
            "null_rate_delta_pp": upstream_null_delta,
        },
        "row_count_pct_change": pct_change,
        "z_score": z_score,
        "checks": checks,
        "anomaly_type": "+".join(anomaly_parts) if anomaly_parts else "NONE",
        "incident_id": None,
        "evaluated_at": utc_now(),
    }

    if not failed:
        return verdict

    summary = (
        f"{table_name} for run_day {run_day} breached {len(failed)} guardrail(s): "
        f"{row_count:,} rows ({pct_change:+.1f}% vs {baseline['window_days']}-day mean "
        f"{mean_rows:,.0f}, z={z_score:+.2f}); "
        f"{upstream_table} NULL registration_date at "
        f"{float(upstream_metrics['null_rate']):.2f}% "
        f"(baseline {float(upstream_baseline['mean_null_rate']):.2f}%). "
        f"Pipeline job {JOB_NAME} still reported SUCCESS - silent data failure."
    )
    incident_id = _upsert_incident(
        conn,
        table_name=table_name,
        run_day=run_day,
        severity="CRITICAL" if (drop_triggered or not checks[3]["passed"]) else "WARNING",
        anomaly_type=verdict["anomaly_type"],
        summary=summary,
    )
    verdict["incident_id"] = incident_id
    verdict["summary"] = summary
    return verdict


def recovery_is_confirmed(
    expected_rows: int | None = None,
    run_day: int = 15,
    table_name: str = DEFAULT_TABLE,
    tolerance: float = 0.01,
) -> dict[str, Any]:
    """Post-fix verification used by the closed-loop approval endpoint.

    The repaired partition is compared with what actually arrived upstream for
    the same run day (rows in should equal rows out for this transform). The
    14-day mean is only a fallback, because a healthy day can legitimately sit
    more than 1% away from the historical average.
    """
    with connection() as conn:
        metrics = record_metrics(table_name, run_day, conn=conn)
        baseline = baseline_stats(table_name, run_day, conn=conn)
        verdict = _evaluate(conn, table_name, run_day, DEFAULT_UPSTREAM)
        upstream = conn.execute(
            f"SELECT COUNT(*) AS c FROM {DEFAULT_UPSTREAM} WHERE run_day = ?", (run_day,)
        ).fetchone()
    upstream_rows = int(upstream["c"]) if upstream else 0
    if expected_rows is not None:
        target = int(expected_rows)
    elif upstream_rows > 0:
        target = upstream_rows
    else:
        target = int(baseline["mean_row_count"])
    rows = int(metrics["row_count"])
    within_tolerance = target > 0 and abs(rows - target) / target <= tolerance
    return {
        "table_name": table_name,
        "run_day": run_day,
        "row_count": rows,
        "expected_row_count": target,
        "upstream_row_count": upstream_rows,
        "baseline_mean": int(round(float(baseline["mean_row_count"]))),
        "within_tolerance": bool(within_tolerance),
        "recovered": bool(within_tolerance and verdict["status"] == "PASS"),
        "verdict": verdict,
        "checked_at": utc_now(),
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    """Guard helper kept public for UI-side scorecards."""
    if not denominator:
        return 0.0
    return round(numerator / denominator, 4)


def zscore_of(value: float, mean: float, std: float) -> float:
    """Helper mirroring the guardrail maths for dashboards."""
    if not std:
        return 0.0 if value >= mean else -99.0
    return round((value - mean) / std, 2)


def is_math_finite(value: float) -> bool:
    """Small utility so NaN/Inf never leak into JSON responses."""
    return isinstance(value, (int, float)) and math.isfinite(float(value))
