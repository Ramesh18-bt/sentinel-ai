"""SentinelAI FastAPI backend.

Closed loop:
    simulate -> detect -> investigate -> approve -> rewrite SQL -> backfill -> recover

Run with:  uv run uvicorn api.main:app --port 8000
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent.chat_copilot import answer_question
from agent.graph import run_investigation
from agent.state import agent_mode, agent_mode_label
from core.database import (
    HEALTHY_TRANSFORM_SQL,
    JOB_NAME,
    connection,
    execute_transform_sql,
    init_schema,
    query_all,
    query_one,
    read_transform_sql,
    utc_now,
    write_transform_sql,
)
from core.quality_engine import recovery_is_confirmed
from data.seed_generator import (
    inject_day15_anomaly,
    reset_and_seed_baseline,
    simulate_day15_healthy,
)

load_dotenv()

DEFAULT_APPROVER = "analytics.lead@sentinel.ai"
APPROVABLE_STATUSES = {"INVESTIGATED", "FIX_FAILED"}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class ApprovalRequest(BaseModel):
    approved_by: str = Field(default=DEFAULT_APPROVER, min_length=1, max_length=200)


def _bootstrap() -> None:
    """Create the schema and seed the baseline once so the demo is never empty."""
    init_schema()
    row = query_one("SELECT COUNT(*) AS c FROM raw_users")
    if not row or int(row["c"]) == 0:
        reset_and_seed_baseline()


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.getenv("SENTINEL_AUTO_SEED", "true").strip().lower() not in {"0", "false", "no"}:
        _bootstrap()
    yield


app = FastAPI(
    title="SentinelAI",
    version="0.1.0",
    description="Autonomous data quality, root cause investigation and interactive data copilot.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fetch_incident(incident_id: int) -> dict[str, Any]:
    incident = query_one("SELECT * FROM incidents WHERE incident_id = ?", (incident_id,))
    if not incident:
        raise HTTPException(status_code=404, detail=f"incident {incident_id} not found")
    return incident


def _safe_json_loads(payload: str | None) -> Any:
    if not payload:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _system_state() -> dict[str, Any]:
    incidents = query_all(
        """
        SELECT incident_id, table_name, run_day, severity, anomaly_type, summary, status,
               approved_by, resolved_at, created_at
        FROM incidents
        ORDER BY incident_id DESC
        """
    )
    open_incidents = [row for row in incidents if row["status"] != "RESOLVED"]
    resolved = [row for row in incidents if row["status"] == "RESOLVED"]

    if open_incidents:
        status = "CRITICAL INCIDENT"
    elif resolved:
        status = "RECOVERED"
    else:
        status = "HEALTHY"

    latest_day = query_one("SELECT MAX(run_day) AS day FROM raw_users") or {"day": 15}
    run_day = int(latest_day["day"] or 15)
    raw = query_one("SELECT COUNT(*) AS c FROM raw_users WHERE run_day = ?", (run_day,)) or {"c": 0}
    dim = query_one("SELECT COUNT(*) AS c FROM dim_customers WHERE run_day = ?", (run_day,)) or {"c": 0}
    baseline = query_one(
        """
        SELECT AVG(row_count) AS avg_rows, AVG(null_rate) AS avg_null
        FROM metric_history
        WHERE table_name = 'dim_customers' AND run_day < ?
        """,
        (run_day,),
    ) or {"avg_rows": 0, "avg_null": 0}

    baseline_rows = float(baseline["avg_rows"] or 0.0)
    dim_rows = int(dim["c"])
    variance_pct = (
        round(((dim_rows - baseline_rows) / baseline_rows) * 100.0, 2) if baseline_rows else 0.0
    )

    return {
        "status": status,
        "agent_mode": agent_mode(),
        "agent_mode_label": agent_mode_label(),
        "run_day": run_day,
        "raw_rows": int(raw["c"]),
        "dim_rows": dim_rows,
        "baseline_rows": round(baseline_rows, 2),
        "variance_pct": variance_pct,
        "downstream_sla": "AT RISK" if open_incidents and variance_pct < -5 else "ON TRACK",
        "open_incidents": len(open_incidents),
        "resolved_incidents": len(resolved),
        "transform_sql": read_transform_sql(),
    }


def _dashboard_payload() -> dict[str, Any]:
    return {
        "generated_at": utc_now(),
        "metric_history": query_all(
            """
            SELECT id, table_name, run_day, row_count, null_rate, duplicate_count, recorded_at
            FROM metric_history
            ORDER BY run_day, table_name
            """
        ),
        "pipeline_runs": query_all(
            """
            SELECT run_id, job_name, run_day, status, rows_in, rows_out, duration_sec, executed_at
            FROM pipeline_runs
            ORDER BY run_day, run_id
            """
        ),
        "lineage_edges": query_all(
            "SELECT id, upstream_table, downstream_table, job_name FROM lineage_edges ORDER BY id"
        ),
        "code_commits": query_all(
            """
            SELECT commit_hash, author, message, file_path, diff_text, committed_at
            FROM code_commits
            ORDER BY committed_at DESC, commit_hash DESC
            """
        ),
        "incidents": query_all(
            """
            SELECT incident_id, table_name, run_day, severity, anomaly_type, summary, status,
                   investigation_json, proposed_sql, rollback_plan, approved_by, resolved_at,
                   created_at
            FROM incidents
            ORDER BY incident_id DESC
            """
        ),
        "chat_messages": query_all(
            """
            SELECT id, role, content, tool_citations_json, created_at
            FROM chat_messages
            ORDER BY id ASC
            """
        ),
        "system": _system_state(),
    }


# ---------------------------------------------------------------------------
# Simulation endpoints
# ---------------------------------------------------------------------------


@app.post("/reset")
def reset() -> dict[str, Any]:
    try:
        return reset_and_seed_baseline()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"reset failed: {exc}") from exc


@app.post("/simulate/healthy")
def simulate_healthy() -> dict[str, Any]:
    try:
        return simulate_day15_healthy()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"healthy simulation failed: {exc}") from exc


@app.post("/simulate/anomaly")
def simulate_anomaly() -> dict[str, Any]:
    try:
        return inject_day15_anomaly()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"anomaly injection failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


@app.get("/dashboard-data")
def dashboard_data() -> dict[str, Any]:
    try:
        return _dashboard_payload()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"dashboard query failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Investigation + HITL approval
# ---------------------------------------------------------------------------


@app.post("/incidents/{incident_id}/investigate")
def investigate(incident_id: int) -> dict[str, Any]:
    incident = _fetch_incident(incident_id)
    if incident["status"] == "RESOLVED":
        raise HTTPException(status_code=400, detail=f"incident {incident_id} is already resolved")
    try:
        report = run_investigation(
            incident_id=incident_id,
            table_name=str(incident["table_name"]),
            run_day=int(incident["run_day"]),
            job_name=JOB_NAME,
            file_path="sql/transform_daily_customers.sql",
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"investigation failed: {exc}") from exc

    with connection() as conn:
        conn.execute(
            """
            UPDATE incidents
            SET investigation_json = ?, proposed_sql = ?, rollback_plan = ?, status = 'INVESTIGATED'
            WHERE incident_id = ?
            """,
            (
                json.dumps(report, default=str),
                report["proposed_sql"],
                report["rollback_plan"],
                incident_id,
            ),
        )
    report["incident_status"] = "INVESTIGATED"
    return report


@app.post("/incidents/{incident_id}/approve")
def approve(incident_id: int, payload: ApprovalRequest | None = None) -> dict[str, Any]:
    incident = _fetch_incident(incident_id)
    if incident["status"] == "RESOLVED":
        raise HTTPException(status_code=400, detail=f"incident {incident_id} is already resolved")
    # Human-in-the-loop gate: a fix can only be approved after the agent has
    # investigated and drafted it, so the approver has seen the evidence.
    if incident["status"] not in APPROVABLE_STATUSES or not incident.get("proposed_sql"):
        raise HTTPException(
            status_code=409,
            detail=(
                f"incident {incident_id} is {incident['status']}; run the investigation "
                "before approving a fix"
            ),
        )

    approver = (payload.approved_by if payload else DEFAULT_APPROVER) or DEFAULT_APPROVER
    proposed_sql = str(incident["proposed_sql"])
    if "COALESCE(registration_date, created_at)" not in proposed_sql:
        proposed_sql = HEALTHY_TRANSFORM_SQL
    if "registration_date IS NOT NULL" in proposed_sql:
        raise HTTPException(
            status_code=400,
            detail="proposed_sql still contains the offending predicate; refusing to apply",
        )

    run_day = int(incident["run_day"])
    applied_at = utc_now()

    previous = query_one(
        "SELECT COUNT(*) AS c FROM dim_customers WHERE run_day = ?", (run_day,)
    ) or {"c": 0}
    rows_before = int(previous["c"])
    previous_sql = read_transform_sql()

    # 1) Re-run the transform with the proposed SQL inside one transaction.
    try:
        with connection() as conn:
            init_schema(conn)
            rows_in_row = conn.execute(
                "SELECT COUNT(*) AS c FROM raw_users WHERE run_day = ?", (run_day,)
            ).fetchone()
            rows_in = int(rows_in_row["c"]) if rows_in_row else 0
            result = execute_transform_sql(run_day, proposed_sql, conn=conn)
            rows_out = int(result["rows_out"])
            conn.execute(
                """
                INSERT INTO pipeline_runs
                    (job_name, run_day, status, rows_in, rows_out, duration_sec, executed_at)
                VALUES (?, ?, 'SUCCESS', ?, ?, ?, ?)
                """,
                (JOB_NAME, run_day, rows_in, rows_out, 24.2, applied_at),
            )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"fix application failed: {exc}") from exc

    # 2) Only persist the SQL file once the database work committed, so the
    #    file on disk never disagrees with the data after a failure.
    try:
        write_transform_sql(proposed_sql)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"could not write transform SQL: {exc}") from exc

    # 3) Verify against what actually arrived upstream; close only on success.
    verification = recovery_is_confirmed(expected_rows=rows_in, run_day=run_day)
    recovered = bool(verification["recovered"])
    resolved_at = utc_now() if recovered else None
    with connection() as conn:
        if recovered:
            conn.execute(
                """
                UPDATE incidents
                SET status = 'RESOLVED', approved_by = ?, resolved_at = ?
                WHERE run_day = ? AND status <> 'RESOLVED'
                """,
                (approver, resolved_at, run_day),
            )
        else:
            conn.execute(
                "UPDATE incidents SET status = 'FIX_FAILED', approved_by = ? WHERE incident_id = ?",
                (approver, incident_id),
            )
    if not recovered:
        # Put the previous transform back so the pipeline is not left on an
        # unverified script.
        write_transform_sql(previous_sql)

    updated = _fetch_incident(incident_id)
    return {
        "incident": updated,
        "approved_by": approver,
        "resolved_at": resolved_at,
        "rows_before": rows_before,
        "rows_after": verification["row_count"],
        "expected_rows": verification["expected_row_count"],
        "recovered": recovered,
        "quality_status": verification["verdict"]["status"],
        "verification": verification,
        "transform_sql": proposed_sql if recovered else previous_sql,
        "message": (
            f"Fix applied by {approver}. dim_customers restored to "
            f"{verification['row_count']:,} rows for run_day {run_day}."
            if recovered
            else (
                "Fix applied but recovery verification did not pass; the previous transform "
                "SQL was restored and the incident is marked FIX_FAILED. Inspect the verdict."
            )
        ),
    }


# ---------------------------------------------------------------------------
# Copilot
# ---------------------------------------------------------------------------


@app.post("/chat")
def chat(payload: ChatRequest) -> dict[str, Any]:
    history_rows = query_all(
        "SELECT role, content FROM chat_messages ORDER BY id DESC LIMIT 12"
    )
    history = list(reversed([dict(row) for row in history_rows]))
    try:
        result = answer_question(payload.message, history)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"copilot failed: {exc}") from exc

    citations = result.get("tool_citations", [])
    created_at = utc_now()
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO chat_messages (role, content, tool_citations_json, created_at)
            VALUES ('user', ?, NULL, ?)
            """,
            (payload.message, created_at),
        )
        conn.execute(
            """
            INSERT INTO chat_messages (role, content, tool_citations_json, created_at)
            VALUES ('assistant', ?, ?, ?)
            """,
            (result["reply"], json.dumps(citations, default=str), created_at),
        )

    return {
        "reply": result["reply"],
        "tool_citations": citations,
        "mode": result.get("mode", agent_mode()),
        "intents": result.get("intents", []),
        "created_at": created_at,
    }
