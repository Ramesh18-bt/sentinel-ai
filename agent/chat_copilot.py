"""SentinelAI Data Copilot.

Answers pipeline-health questions and always grounds the reply in live SQLite
facts. Offline mode routes intent deterministically and calls the same
``core.tools`` functions the agent uses; live mode binds those tools to
``ChatOpenAI(gpt-4o-mini, temperature=0.1)``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agent.state import COPILOT_MODEL, TRANSFORM_SQL_PATH, agent_mode, is_live_mode
from core.database import JOB_NAME, query_all, query_one
from core.tools import FORENSIC_TOOLS, invoke_tool

SYSTEM_PROMPT = (
    "You are SentinelAI's Data Copilot, a senior analytics engineer embedded in the "
    "reliability platform. Answer using ONLY the read-only forensic tools. Cite concrete "
    "numbers, commit hashes and downstream assets. If a tool returns an error, say so "
    "plainly. Be concise and operational."
)

INTENT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("git", ("git", "commit", "diff", "a8f92c1", "code change", "blame", "revert")),
    ("downstream", ("downstream", "dashboard", "impact", "blast", "consumer", "revenue", "sla", "lineage", "upstream")),
    ("nulls", ("null", "guest", "registration_date", "missing", "nulls", "coalesce", "timestamp")),
    ("rows", ("row", "count", "volume", "drop", "62400", "62,400", "101200", "101,200", "shr", "rows_out")),
    ("logs", ("job", "status", "success", "log", "duration", "pipeline_run", "orchestrat")),
    ("fix", ("fix", "propose", "proposed", "sql", "rollback", "approve", "remediat", "repair")),
    ("incident", ("incident", "severity", "alert", "health", "status", "resolved", "recovered")),
    ("root", ("why", "root cause", "rca", "cause", "explain", "reason")),
)


# ---------------------------------------------------------------------------
# Tool access with per-turn caching
# ---------------------------------------------------------------------------


class _Session:
    """Caches tool output within one chat turn and records citations."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], dict[str, Any]] = {}
        self.citations: list[dict[str, Any]] = []
        self.outputs: dict[str, dict[str, Any]] = {}

    def call(self, name: str, **arguments: Any) -> dict[str, Any]:
        key = (name, json.dumps(arguments, sort_keys=True, default=str))
        if key in self._cache:
            return self._cache[key]
        try:
            output = invoke_tool(name, arguments)
        except Exception as exc:  # noqa: BLE001 - surfaced in the reply
            output = {"tool": name, "ok": False, "error": str(exc)}
        self._cache[key] = output
        self.citations.append(
            {"tool": name, "arguments": arguments, "ok": bool(output.get("ok", False))}
        )
        self.outputs.setdefault(_alias(name), output)
        return output


def _alias(tool_name: str) -> str:
    return {
        "inspect_lineage": "lineage",
        "profile_upstream_table": "profile",
        "query_job_logs": "logs",
        "get_git_diff": "git",
        "run_readonly_diagnostic_sql": "sql",
    }.get(tool_name, tool_name)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _latest_incident() -> dict[str, Any] | None:
    return query_one(
        """
        SELECT incident_id, table_name, run_day, severity, anomaly_type, summary,
               status, approved_by, resolved_at, created_at
        FROM incidents
        ORDER BY (status = 'RESOLVED'), incident_id DESC
        LIMIT 1
        """
    )


def _metric_pair(run_day: int = 15) -> dict[str, Any]:
    raw = query_one(
        "SELECT COUNT(*) AS c FROM raw_users WHERE run_day = ?", (run_day,)
    ) or {"c": 0}
    dim = query_one(
        "SELECT COUNT(*) AS c FROM dim_customers WHERE run_day = ?", (run_day,)
    ) or {"c": 0}
    baseline = query_one(
        """
        SELECT AVG(row_count) AS avg_rows, AVG(null_rate) AS avg_null
        FROM metric_history
        WHERE table_name = 'dim_customers' AND run_day < ?
        """,
        (run_day,),
    ) or {"avg_rows": 0, "avg_null": 0}
    raw_baseline = query_one(
        """
        SELECT AVG(null_rate) AS avg_null
        FROM metric_history
        WHERE table_name = 'raw_users' AND run_day < ?
        """,
        (run_day,),
    ) or {"avg_null": 0}
    return {
        "raw_rows": int(raw["c"]),
        "dim_rows": int(dim["c"]),
        "baseline_rows": _num(baseline.get("avg_rows")),
        "baseline_null": _num(raw_baseline.get("avg_null")),
    }


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------


def detect_intents(message: str) -> list[str]:
    """Keyword-routed intents, ordered and de-duplicated."""
    text = (message or "").lower()
    hits: list[str] = []
    for intent, keywords in INTENT_RULES:
        if any(re.search(rf"\b{re.escape(word)}", text) for word in keywords):
            hits.append(intent)
    if "root" not in hits and any(word in text for word in ("why", "what happened", "explain")):
        hits.insert(0, "root")
    if not hits:
        hits = ["overview"]
    return hits


# ---------------------------------------------------------------------------
# Offline responder
# ---------------------------------------------------------------------------


def _offline_answer(message: str) -> dict[str, Any]:
    session = _Session()
    intents = detect_intents(message)
    run_day = 15
    if re.search(r"day\s*(\d{1,2})", message.lower()):
        run_day = max(1, min(15, int(re.search(r"day\s*(\d{1,2})", message.lower()).group(1))))

    snapshot = _metric_pair(run_day)
    lineage = session.call("inspect_lineage", table_name="dim_customers")
    profile = session.call(
        "profile_upstream_table", table_name="raw_users", run_day=run_day
    )
    logs = session.call("query_job_logs", job_name=JOB_NAME, run_day=run_day)
    git = session.call("get_git_diff", file_path=TRANSFORM_SQL_PATH)
    incident = _latest_incident()

    rows_in = int(logs.get("rows_in") or snapshot["raw_rows"] or 0)
    rows_out = int(logs.get("rows_out") or snapshot["dim_rows"] or 0)
    baseline_rows = _num(profile.get("baseline_14d_avg_row_count"), snapshot["baseline_rows"]) or 100_000.0
    null_rate = _num(profile.get("null_rate"), 0.0)
    baseline_null = _num(profile.get("baseline_14d_avg_null_rate"), snapshot["baseline_null"])
    drop_pct = round(((rows_out - baseline_rows) / baseline_rows) * 100.0, 2) if baseline_rows else 0.0
    commit_hash = str(git.get("latest_commit") or "n/a")
    author = str(git.get("latest_author") or "n/a")
    diff_text = str(git.get("diff_text") or "").strip()
    assets = ", ".join(str(a) for a in (lineage.get("impacted_assets") or ["exec_revenue_dashboard"]))
    status = str(logs.get("status") or "UNKNOWN")
    severity = str((incident or {}).get("severity") or "NONE")
    incident_status = str((incident or {}).get("status") or "NONE")

    blocks: list[str] = []

    if "overview" in intents or len(intents) > 3:
        blocks.append(
            f"**Pipeline snapshot (run_day {run_day})**\n"
            f"- `raw_users`: **{rows_in:,}** rows ingested\n"
            f"- `dim_customers`: **{rows_out:,}** rows materialised "
            f"({drop_pct:+.1f}% vs the {int(profile.get('baseline_window_days') or 14)}-day mean of {baseline_rows:,.0f})\n"
            f"- Guest checkouts with NULL `registration_date`: **{null_rate:.2f}%** "
            f"(baseline {baseline_null:.2f}%)\n"
            f"- Job `{JOB_NAME}`: **{status}** in {logs.get('duration_sec', 0)}s\n"
            f"- Incident state: **{severity} / {incident_status}**"
        )

    if "root" in intents or "git" in intents:
        blocks.append(
            f"**Root cause**\n"
            f"Commit `{commit_hash}` by **{author}** changed `{TRANSFORM_SQL_PATH}`:\n"
            f"```diff\n{diff_text}\n```\n"
            f"That predicate removes every guest checkout before it reaches `dim_customers`. "
            f"Combined with the promo NULL spike of {null_rate:.2f}% vs {baseline_null:.2f}% baseline, "
            f"the transform emitted **{rows_out:,}** rows from **{rows_in:,}** ingested "
            f"({max(rows_in - rows_out, 0):,} rows silently filtered)."
        )

    if "downstream" in intents:
        blocks.append(
            f"**Downstream blast radius**\n"
            f"- Impacted assets: **{assets}**\n"
            f"- Impacted jobs: {', '.join(str(j) for j in (lineage.get('impacted_jobs') or ['build_exec_mart']))}\n"
            f"- Upstream sources: {', '.join(str(u['upstream_table']) for u in (lineage.get('upstream_sources') or []))}\n"
            f"- `exec_revenue_dashboard` is reporting revenue off a "
            f"{abs(drop_pct):.1f}% incomplete customer base until the fix lands."
        )

    if "rows" in intents:
        daily = query_all(
            """
            SELECT r.run_day, r.c AS raw_rows, COALESCE(d.c, 0) AS dim_rows
            FROM (SELECT run_day, COUNT(*) AS c FROM raw_users GROUP BY run_day) r
            LEFT JOIN (SELECT run_day, COUNT(*) AS c FROM dim_customers GROUP BY run_day) d
              ON d.run_day = r.run_day
            WHERE r.run_day BETWEEN ? AND ?
            ORDER BY r.run_day
            """,
            (max(1, run_day - 3), run_day),
        )
        detail = "\n".join(
            f"- Day {row['run_day']}: raw {int(row['raw_rows']):,} -> dim {int(row['dim_rows']):,}"
            for row in daily
        )
        blocks.append(f"**Row counts (last 4 partitions)**\n{detail}")

    if "nulls" in intents:
        diag = session.call(
            "run_readonly_diagnostic_sql",
            query=(
                "SELECT run_day, COUNT(*) AS rows, "
                "SUM(CASE WHEN registration_date IS NULL THEN 1 ELSE 0 END) AS nulls, "
                "ROUND(100.0 * SUM(CASE WHEN registration_date IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS null_pct "
                "FROM raw_users WHERE run_day >= 13 GROUP BY run_day ORDER BY run_day"
            ),
        )
        series = diag.get("rows") or []
        detail = "\n".join(
            f"- Day {row['run_day']}: {row['null_pct']}% NULL "
            f"({int(row['nulls']):,} of {int(row['rows']):,})"
            for row in series
        )
        blocks.append(
            f"**NULL `registration_date` comparison**\n{detail}\n"
            f"14-day baseline: **{baseline_null:.2f}%**. Day {run_day} is "
            f"**{null_rate - baseline_null:+.2f}pp** above baseline."
        )

    if "logs" in intents:
        blocks.append(
            f"**Job telemetry**\n"
            f"- `{JOB_NAME}` run_day {run_day}: status **{status}**, rows_in **{rows_in:,}**, "
            f"rows_out **{rows_out:,}**, {logs.get('duration_sec', 0)}s\n"
            f"- Silent failure detected: **{logs.get('silent_failure')}** "
            f"({logs.get('dropped_rows', max(rows_in - rows_out, 0)):,} rows unaccounted for)"
        )

    if "fix" in intents:
        blocks.append(
            "**Safe remediation**\n"
            "```sql\n"
            "DELETE FROM dim_customers WHERE run_day = :run_day;\n\n"
            "INSERT INTO dim_customers (user_id, run_day, email, effective_reg_date)\n"
            "SELECT user_id, run_day, email, COALESCE(registration_date, created_at)\n"
            "FROM raw_users\n"
            "WHERE run_day = :run_day;\n"
            "```\n"
            f"Rollback: `git revert {commit_hash}`, re-run Day {run_day}, assert "
            f"`dim_customers` is within 1% of {baseline_rows:,.0f} rows, then refresh "
            f"{assets}."
        )

    if "incident" in intents and incident:
        blocks.append(
            f"**Incident #{incident['incident_id']}** - {incident['severity']} "
            f"({incident['status']}) on `{incident['table_name']}` run_day {incident['run_day']}.\n"
            f"{incident['summary']}"
        )

    if not blocks:
        blocks.append(
            f"`dim_customers` currently holds **{rows_out:,}** rows for run_day {run_day} "
            f"against a {baseline_rows:,.0f} baseline. Ask me about the root cause, the "
            f"downstream impact, the git diff, or the exact NULL-rate comparison."
        )

    reply = "\n\n".join(blocks)
    return {
        "reply": reply,
        "tool_citations": session.citations,
        "mode": "offline",
        "intents": intents,
    }


# ---------------------------------------------------------------------------
# Live responder
# ---------------------------------------------------------------------------


def _history_messages(history: list[dict[str, Any]] | None) -> list[Any]:
    from langchain_core.messages import AIMessage, HumanMessage

    messages: list[Any] = []
    for item in history or []:
        role = str(item.get("role", "user"))
        content = str(item.get("content", ""))
        if not content.strip():
            continue
        messages.append(AIMessage(content=content) if role == "assistant" else HumanMessage(content=content))
    return messages


def _live_answer(message: str, history: list[dict[str, Any]] | None) -> dict[str, Any]:
    from langchain_core.messages import SystemMessage, ToolMessage
    from langchain_openai import ChatOpenAI

    session = _Session()
    llm = ChatOpenAI(model=COPILOT_MODEL, temperature=0.1).bind_tools(FORENSIC_TOOLS)
    messages: list[Any] = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(_history_messages(history))
    messages.append(_history_user(message))

    reply_text = ""
    try:
        for _ in range(6):
            response = llm.invoke(messages)
            messages.append(response)
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                reply_text = str(response.content or "").strip()
                break
            for call in tool_calls:
                name = call.get("name", "")
                args = call.get("args", {}) or {}
                output = session.call(name, **args)
                messages.append(
                    ToolMessage(
                        content=json.dumps(output, default=str),
                        tool_call_id=call.get("id", f"call-{len(session.citations)}"),
                    )
                )
    except Exception as exc:  # noqa: BLE001 - degrade to the deterministic engine
        fallback = _offline_answer(message)
        fallback["reply"] = (
            f"_Live agent unavailable ({exc}). Deterministic engine answer:_\n\n"
            + str(fallback["reply"])
        )
        fallback["mode"] = "offline-fallback"
        return fallback

    if not reply_text:
        reply_text = (
            "I gathered forensic evidence but could not synthesise a final answer. "
            f"Latest job status: {(session.outputs.get('logs') or {}).get('status', 'UNKNOWN')}."
        )

    return {
        "reply": reply_text,
        "tool_citations": session.citations,
        "mode": "live",
        "intents": detect_intents(message),
    }


def _history_user(message: str) -> Any:
    from langchain_core.messages import HumanMessage

    return HumanMessage(content=message)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def answer_question(
    message: str,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Chat entrypoint used by ``POST /chat`` and the Streamlit copilot tab."""
    clean = (message or "").strip()
    if not clean:
        return {
            "reply": "Ask me about pipeline health, the Day 15 drop, downstream impact, or the git diff.",
            "tool_citations": [],
            "mode": agent_mode(),
            "intents": [],
        }
    if is_live_mode():
        return _live_answer(clean, history)
    return _offline_answer(clean)


def suggested_prompts() -> list[str]:
    """Quick-prompt chips rendered by the UI."""
    return [
        "Why did dim_customers drop on Day 15?",
        "Which dashboards are impacted?",
        "Show me the exact git diff causing this",
        "Compare Day 15 null rate vs 14-day baseline",
        "What job telemetry did the pipeline emit?",
        "What is the safe fix and rollback plan?",
    ]


__all__ = ["answer_question", "detect_intents", "suggested_prompts"]
