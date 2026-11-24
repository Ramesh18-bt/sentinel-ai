"""SentinelAI investigation agent.

Graph:  gather_evidence -> analyze_and_rank -> draft_safe_fix

* Offline demo mode (default): invokes the real ``@tool`` forensic functions
  against ``sentinel.db`` and ranks causes from those live numbers.
* Live mode (``SENTINEL_MOCK_MODE=false`` + ``OPENAI_API_KEY``): binds the same
  tools to ``ChatOpenAI(gpt-4o-mini, temperature=0)`` for the evidence sweep and
  validates structured LLM output before assembling the final report.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from agent.state import (
    AGENT_MODEL,
    TRANSFORM_SQL_PATH,
    AgentState,
    CauseAnalysis,
    FixPlan,
    InvestigationReport,
    RankedCause,
    ToolTraceEntry,
    agent_mode,
    is_live_mode,
)
from core.database import HEALTHY_TRANSFORM_SQL, JOB_NAME, PROJECT_ROOT, read_transform_sql, utc_now
from core.tools import FORENSIC_TOOLS, invoke_tool

SUPPORTED_TABLES = ("dim_customers", "raw_users", "stg_users_cleaned")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return utc_now()


def _invoke(tool_name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], ToolTraceEntry]:
    from time import perf_counter

    started = perf_counter()
    try:
        output = invoke_tool(tool_name, arguments)
    except Exception as exc:  # noqa: BLE001 - unknown tool or bad arguments
        output = {"tool": tool_name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(output, dict):
        output = {"tool": tool_name, "ok": False, "error": f"unexpected tool output: {output!r}"[:500]}
    duration_ms = round((perf_counter() - started) * 1000, 2)
    entry = ToolTraceEntry(
        tool=tool_name,
        arguments=arguments,
        ok=bool(output.get("ok", False)),
        output=output,
        duration_ms=duration_ms,
    )
    return output, entry


def _diagnostic_query(run_day: int) -> str:
    return (
        "SELECT run_day, COUNT(*) AS raw_rows, "
        "SUM(CASE WHEN registration_date IS NULL THEN 1 ELSE 0 END) AS null_registrations, "
        "ROUND(100.0 * SUM(CASE WHEN registration_date IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) "
        "AS null_pct "
        "FROM raw_users "
        f"WHERE run_day BETWEEN {max(1, run_day - 3)} AND {run_day} "
        "GROUP BY run_day ORDER BY run_day"
    )


def _deterministic_plan(state: AgentState) -> list[tuple[str, dict[str, Any]]]:
    table_name = state.get("table_name", "dim_customers")
    run_day = int(state.get("run_day", 15))
    job_name = state.get("job_name", JOB_NAME)
    file_path = state.get("file_path", TRANSFORM_SQL_PATH)
    return [
        ("inspect_lineage", {"table_name": table_name}),
        ("profile_upstream_table", {"table_name": "raw_users", "run_day": run_day}),
        ("query_job_logs", {"job_name": job_name, "run_day": run_day}),
        ("get_git_diff", {"file_path": file_path}),
        ("run_readonly_diagnostic_sql", {"query": _diagnostic_query(run_day)}),
    ]


def _trace_key(tool_name: str) -> str:
    return {
        "inspect_lineage": "lineage",
        "profile_upstream_table": "profile_upstream",
        "query_job_logs": "job_logs",
        "get_git_diff": "git_diff",
        "run_readonly_diagnostic_sql": "diagnostic_sql",
    }.get(tool_name, tool_name)


def _evidence_from_trace(
    trace: list[ToolTraceEntry],
    plan: list[tuple[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Map tool output to evidence keys, preferring the planned calls.

    The analysis expects e.g. ``profile_upstream`` to describe raw_users; if
    the live agent also profiled another table, the planned call wins.
    """
    planned = {
        (name, json.dumps(args, sort_keys=True, default=str)) for name, args in (plan or [])
    }
    evidence: dict[str, Any] = {}
    preferred: set[str] = set()
    for entry in trace:
        output = entry.output if isinstance(entry.output, dict) else {}
        key = _trace_key(entry.tool)
        is_planned = (entry.tool, json.dumps(entry.arguments, sort_keys=True, default=str)) in planned
        if key in preferred and not is_planned:
            continue
        if key in evidence and not entry.ok and not is_planned:
            continue
        evidence[key] = output
        if is_planned:
            preferred.add(key)
    return evidence


def _rows(state_rows: dict[str, Any], evidence: dict[str, Any]) -> tuple[int, int]:
    job = evidence.get("job_logs", {}) or {}
    profile = evidence.get("profile_upstream", {}) or {}
    rows_in = int(job.get("rows_in") or profile.get("row_count") or state_rows.get("rows_in") or 0)
    rows_out = int(job.get("rows_out") or 0)
    return rows_in, rows_out


# ---------------------------------------------------------------------------
# Node 1: gather_evidence
# ---------------------------------------------------------------------------


def gather_evidence(state: AgentState) -> dict[str, Any]:
    """Run the read-only forensic sweep and capture every raw JSON payload."""
    trace: list[ToolTraceEntry] = []

    if is_live_mode():
        trace.extend(_live_tool_sweep(state))

    # Only skip a planned call when the live agent already made the identical
    # call successfully; failed or differently-scoped calls are re-run so the
    # analysis always sees the evidence it expects.
    collected = {
        (entry.tool, json.dumps(entry.arguments, sort_keys=True, default=str))
        for entry in trace
        if entry.ok
    }
    for tool_name, arguments in _deterministic_plan(state):
        if (tool_name, json.dumps(arguments, sort_keys=True, default=str)) in collected:
            continue
        _, entry = _invoke(tool_name, arguments)
        trace.append(entry)

    evidence = _evidence_from_trace(trace, _deterministic_plan(state))
    rows_in, rows_out = _rows({}, evidence)
    return {
        "mode": agent_mode(),
        "tool_trace": [entry.model_dump() for entry in trace],
        "evidence": evidence,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "error": "",
    }


def _live_tool_sweep(state: AgentState) -> list[ToolTraceEntry]:
    """Let the bound LLM decide which forensic tools to call (best effort)."""
    trace: list[ToolTraceEntry] = []
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=AGENT_MODEL, temperature=0).bind_tools(FORENSIC_TOOLS)
        messages: list[Any] = [
            SystemMessage(
                content=(
                    "You are SentinelAI, an autonomous data-reliability investigator. "
                    "Use the read-only forensic tools to gather evidence about the failing "
                    "table and run day. Call the most relevant tools, then stop."
                )
            ),
            HumanMessage(
                content=(
                    f"Investigate table={state.get('table_name', 'dim_customers')} "
                    f"run_day={state.get('run_day', 15)} "
                    f"job={state.get('job_name', JOB_NAME)}. "
                    "At minimum inspect lineage, profile the upstream raw_users partition, "
                    "read the job logs, and check the most recent git diff for the transform SQL."
                )
            ),
        ]
        for _ in range(5):
            response = llm.invoke(messages)
            messages.append(response)
            tool_calls = getattr(response, "tool_calls", None) or []
            if not tool_calls:
                break
            for call in tool_calls:
                name = call.get("name", "")
                args = call.get("args", {}) or {}
                if name:
                    output, entry = _invoke(name, args)
                    trace.append(entry)
                else:
                    # Every tool call must be answered or the next LLM turn is rejected.
                    output = {"ok": False, "error": "tool call without a name"}
                messages.append(
                    ToolMessage(
                        content=json.dumps(output, default=str),
                        tool_call_id=call.get("id", f"call-{len(trace)}"),
                    )
                )
    except Exception:  # noqa: BLE001 - deterministic plan covers us
        return trace
    return trace


# ---------------------------------------------------------------------------
# Node 2: analyze_and_rank
# ---------------------------------------------------------------------------


def analyze_and_rank(state: AgentState) -> dict[str, Any]:
    """Rank root causes with confidence scores and hard citations."""
    evidence = state.get("evidence", {}) or {}
    if is_live_mode():
        analysis = _live_analysis(state, evidence)
        if analysis is not None:
            return {
                "summary": analysis.summary,
                "ranked_causes": [cause.model_dump() for cause in analysis.ranked_causes],
                "downstream_impact": analysis.downstream_impact
                or _default_downstream(evidence),
                "anomaly_detected": bool(_assess(state, evidence)["anomaly"]),
            }
    summary, causes, impact, anomaly = _deterministic_analysis(state, evidence)
    return {
        "summary": summary,
        "ranked_causes": [cause.model_dump() for cause in causes],
        "downstream_impact": impact,
        "anomaly_detected": anomaly,
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


ANOMALY_DROP_PCT = -15.0
ANOMALY_LOSS_PCT = 2.0
NULL_PREDICATE_RE = re.compile(r"registration_date\s+IS\s+NOT\s+NULL", re.IGNORECASE)


def _assess(state: AgentState, evidence: dict[str, Any]) -> dict[str, Any]:
    """Turn raw tool payloads into the facts every downstream decision uses."""
    profile = evidence.get("profile_upstream", {}) or {}
    job = evidence.get("job_logs", {}) or {}
    git = evidence.get("git_diff", {}) or {}
    lineage = evidence.get("lineage", {}) or {}

    rows_in = int(job.get("rows_in") or profile.get("row_count") or 0)
    rows_out = int(job.get("rows_out") or 0)
    baseline_mean = _safe_float(profile.get("baseline_14d_avg_row_count"), 0.0)
    drop_pct = round(((rows_out - baseline_mean) / baseline_mean) * 100.0, 2) if baseline_mean else 0.0
    row_loss = max(rows_in - rows_out, 0)
    loss_pct = round(row_loss / rows_in * 100.0, 2) if rows_in else 0.0
    null_rate = _safe_float(profile.get("null_rate"), 0.0)
    baseline_null = _safe_float(profile.get("baseline_14d_avg_null_rate"), 0.0)

    suspects = [c for c in (git.get("suspect_commits") or []) if isinstance(c, dict)]
    suspect = suspects[0] if suspects else None
    try:
        sql_has_predicate = bool(NULL_PREDICATE_RE.search(read_transform_sql()))
    except OSError:
        sql_has_predicate = False

    return {
        "table_name": state.get("table_name", "dim_customers"),
        "run_day": int(state.get("run_day", 15)),
        "job_name": state.get("job_name", JOB_NAME),
        "file_path": state.get("file_path", TRANSFORM_SQL_PATH),
        "rows_in": rows_in,
        "rows_out": rows_out,
        "row_loss": row_loss,
        "loss_pct": loss_pct,
        "baseline_mean": baseline_mean,
        "window": int(profile.get("baseline_window_days") or 0),
        "drop_pct": drop_pct,
        "null_rate": null_rate,
        "baseline_null": baseline_null,
        "null_delta": round(null_rate - baseline_null, 2),
        "status": str(job.get("status") or "UNKNOWN"),
        "duration": _safe_float(job.get("duration_sec"), 0.0),
        "suspect": suspect,
        "sql_has_predicate": sql_has_predicate,
        "impacted": [str(a) for a in (lineage.get("impacted_assets") or [])],
        "anomaly": bool(
            (baseline_mean and drop_pct <= ANOMALY_DROP_PCT) or loss_pct > ANOMALY_LOSS_PCT
        ),
    }


def _deterministic_analysis(
    state: AgentState, evidence: dict[str, Any]
) -> tuple[str, list[RankedCause], list[str], bool]:
    facts = _assess(state, evidence)
    t = facts["table_name"]
    day = facts["run_day"]
    rows_in, rows_out = facts["rows_in"], facts["rows_out"]
    mean = facts["baseline_mean"]
    window = facts["window"] or 14
    vs_mean = f"{facts['drop_pct']:+.1f}% vs the {window}-day mean of {mean:,.0f}" if mean else "no baseline yet"

    if not facts["anomaly"]:
        summary = (
            f"No data loss detected for {t} run_day {day}: {rows_out:,} of {rows_in:,} ingested rows "
            f"were materialised ({vs_mean}). Job {facts['job_name']} reported {facts['status']}. "
            f"No remediation is required."
        )
        cause = RankedCause(
            rank=1,
            cause=(
                f"No regression found: {t} row count is within guardrails and every ingested row "
                f"reached the dimension."
            ),
            confidence=0.9,
            citations=[
                f"rows_in {rows_in:,} -> rows_out {rows_out:,}",
                f"{t} {vs_mean}",
                f"NULL registration_date {facts['null_rate']:.2f}% (baseline {facts['baseline_null']:.2f}%)",
            ],
            evidence=[f"pipeline_runs.status = {facts['status']}"],
        )
        return summary, [cause], [], False

    causes: list[RankedCause] = []
    suspect = facts["suspect"]
    impacted = facts["impacted"] or ["exec_revenue_dashboard"]

    if suspect:
        commit_hash = str(suspect.get("commit_hash"))
        author = str(suspect.get("author"))
        diff_text = str(suspect.get("diff_text") or "").strip()
        causes.append(
            RankedCause(
                rank=1,
                cause=(
                    f"Code regression in {facts['file_path']}: commit {commit_hash} ({author}) added "
                    f"`{diff_text.lstrip('+ ').strip()}`, so the transform drops every row with a NULL "
                    f"registration_date (guest checkouts) before it reaches {t}."
                ),
                confidence=0.96 if facts["sql_has_predicate"] else 0.8,
                citations=[
                    f"commit {commit_hash}",
                    f"dev: {author}",
                    f"diff: {diff_text}",
                    f"{t} {rows_out:,} rows ({vs_mean})",
                ]
                + (["predicate present in on-disk transform SQL"] if facts["sql_has_predicate"] else []),
                evidence=[
                    f"Commit message: {suspect.get('message', '')}",
                    f"Job {facts['job_name']} reported {facts['status']} while emitting {rows_out:,} of {rows_in:,} rows",
                ],
            )
        )
    else:
        causes.append(
            RankedCause(
                rank=1,
                cause=(
                    f"Unexplained row loss in {facts['job_name']}: {facts['row_loss']:,} rows "
                    f"({facts['loss_pct']:.1f}%) were filtered between raw_users and {t}, and no recent "
                    f"commit adds a row-dropping NULL filter. Inspect the deployed transform and upstream feed."
                ),
                confidence=0.75 if facts["sql_has_predicate"] else 0.55,
                citations=[
                    f"rows_in {rows_in:,} -> rows_out {rows_out:,}",
                    f"{t} {vs_mean}",
                ]
                + (["NULL predicate present in on-disk transform SQL"] if facts["sql_has_predicate"] else []),
                evidence=[f"recent commits: {[c.get('commit_hash') for c in (evidence.get('git_diff', {}) or {}).get('commits', [])]}"],
            )
        )

    if facts["null_delta"] > 0:
        causes.append(
            RankedCause(
                rank=len(causes) + 1,
                cause=(
                    f"NULL registration_date rose to {facts['null_rate']:.2f}% (14-day baseline "
                    f"{facts['baseline_null']:.2f}%, {facts['null_delta']:+.2f}pp), which amplified the "
                    f"impact of any NULL filter in the transform."
                ),
                confidence=0.74 if facts["null_delta"] > 10 else 0.4,
                citations=[
                    f"{facts['null_rate']:.2f}% NULL rate on raw_users run_day {day}",
                    f"baseline {facts['baseline_null']:.2f}% NULL",
                    f"{facts['row_loss']:,} rows filtered between ingress and {t}",
                ],
                evidence=[
                    f"diagnostic_sql payload: {json.dumps((evidence.get('diagnostic_sql', {}) or {}).get('rows', []))[:600]}"
                ],
            )
        )

    if facts["status"] == "SUCCESS" and facts["row_loss"] > 0:
        job = evidence.get("job_logs", {}) or {}
        causes.append(
            RankedCause(
                rank=len(causes) + 1,
                cause=(
                    f"Silent failure gap: pipeline_runs shows SUCCESS with rows_in {rows_in:,} != "
                    f"rows_out {rows_out:,}, because no row-count reconciliation assertion runs between "
                    f"ingress and mart materialisation."
                ),
                confidence=0.66,
                citations=[
                    "pipeline_runs.status = SUCCESS",
                    f"rows_in {rows_in:,} -> rows_out {rows_out:,}",
                    f"duration_sec {facts['duration']:.1f}",
                    f"downstream consumer {impacted[0]} refreshed with incomplete data",
                ],
                evidence=[f"attempts: {job.get('attempts', [])}"],
            )
        )

    culprit = (
        f"commit {suspect.get('commit_hash')} ({suspect.get('author')}) added a NULL filter to {facts['file_path']}"
        if suspect
        else f"{facts['row_loss']:,} rows were filtered inside {facts['job_name']} with no matching code change"
    )
    summary = (
        f"{t} for run_day {day} collapsed to {rows_out:,} rows ({vs_mean}). "
        f"Most likely cause: {culprit}; NULL registration_date was {facts['null_rate']:.1f}% "
        f"(baseline {facts['baseline_null']:.1f}%). The job still reported {facts['status']}, so "
        f"{', '.join(impacted)} served a customer base missing {facts['loss_pct']:.1f}% of rows."
    )
    return summary, causes, impacted, True


def _default_downstream(evidence: dict[str, Any]) -> list[str]:
    lineage = evidence.get("lineage", {}) or {}
    assets = [str(asset) for asset in (lineage.get("impacted_assets") or [])]
    return assets or ["exec_revenue_dashboard"]


def _live_analysis(state: AgentState, evidence: dict[str, Any]) -> CauseAnalysis | None:
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=AGENT_MODEL, temperature=0).with_structured_output(CauseAnalysis)
        result = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a principal data engineer performing root cause analysis. "
                        "Rank causes by evidence strength. Every cause needs concrete citations "
                        "taken from the supplied forensic evidence (commit hash, author, rows, "
                        "NULL rates, job status). Confidence is between 0 and 1."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Investigate {state.get('table_name')} on run_day {state.get('run_day')} "
                        f"for job {state.get('job_name')}.\n\nForensic evidence JSON:\n"
                        f"{json.dumps(evidence, default=str)[:14000]}"
                    )
                ),
            ]
        )
        if isinstance(result, CauseAnalysis):
            return result
        if isinstance(result, dict):
            return CauseAnalysis.model_validate(result)
    except Exception:  # noqa: BLE001 - deterministic fallback
        return None
    return None


# ---------------------------------------------------------------------------
# Node 3: draft_safe_fix
# ---------------------------------------------------------------------------


def draft_safe_fix(state: AgentState) -> dict[str, Any]:
    """Produce corrected transform SQL plus a concrete rollback plan."""
    evidence = state.get("evidence", {}) or {}
    if is_live_mode() and state.get("anomaly_detected", True):
        plan = _live_fix(state, evidence)
        if plan is not None and _is_safe_sql(plan.proposed_sql):
            return {
                "proposed_sql": plan.proposed_sql,
                "rollback_plan": plan.rollback_plan,
            }
    proposed_sql, rollback_plan = _deterministic_fix(state, evidence)
    return {"proposed_sql": proposed_sql, "rollback_plan": rollback_plan}


def _is_safe_sql(sql_text: str) -> bool:
    if not sql_text or "dim_customers" not in sql_text:
        return False
    if "COALESCE(registration_date, created_at)" not in sql_text:
        return False
    if "registration_date IS NOT NULL" in sql_text:
        return False
    if "DELETE FROM dim_customers WHERE run_day = :run_day" not in sql_text:
        return False
    return True


def _deterministic_fix(state: AgentState, evidence: dict[str, Any]) -> tuple[str, str]:
    facts = _assess(state, evidence)
    table_name = facts["table_name"]
    run_day = facts["run_day"]
    job_name = facts["job_name"]
    file_path = facts["file_path"]
    rows_in = facts["rows_in"]

    if not state.get("anomaly_detected", facts["anomaly"]):
        return HEALTHY_TRANSFORM_SQL, (
            f"NO ACTION REQUIRED - {table_name} run_day {run_day} is healthy "
            f"({facts['rows_out']:,} of {rows_in:,} rows materialised). The proposed SQL is the "
            f"healthy baseline already in use; approving it is a no-op re-run."
        )

    suspect = facts["suspect"]
    revert = (
        f"`git revert {suspect.get('commit_hash')}`"
        if suspect
        else f"restore the last known-good revision of {file_path}"
    )
    rollback_plan = (
        f"ROLLBACK PLAN - incident run_day {run_day} ({table_name})\n"
        f"1. Revert the offending change: {revert} so the transform keeps "
        f"COALESCE(registration_date, created_at) and has no `registration_date IS NOT NULL` filter.\n"
        f"2. Re-run {job_name} for run_day {run_day} with the restored script; expect "
        f"~{rows_in:,} rows in and ~{rows_in:,} rows out.\n"
        f"3. Assert {table_name} row count for run_day {run_day} is within 1% of rows_in "
        f"({rows_in:,}) before publishing.\n"
        f"4. Refresh {', '.join(facts['impacted'] or ['exec_revenue_dashboard'])} for run_day {run_day} "
        f"and confirm it matches the reconciled customer count.\n"
        f"5. If the predicate must stay, apply it in a separate filtered staging model "
        f"(e.g. stg_users_registered) instead of the customer dimension.\n"
        f"6. Add a row-count reconciliation assertion (rows_out >= 0.98 * rows_in) to the "
        f"orchestrator so the next silent drop fails the run.\n"
        f"To undo this fix: restore the previous file contents and re-run step 2."
    )
    return HEALTHY_TRANSFORM_SQL, rollback_plan


def _live_fix(state: AgentState, evidence: dict[str, Any]) -> FixPlan | None:
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=AGENT_MODEL, temperature=0).with_structured_output(FixPlan)
        result = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a principal data engineer writing a safe remediation. "
                        "Return the full corrected SQL script using the named parameter :run_day, "
                        "keeping the COALESCE(registration_date, created_at) fallback and removing "
                        "any `registration_date IS NOT NULL` filter. Also return an ordered, "
                        "executable rollback plan."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Current incident: {json.dumps(evidence, default=str)[:12000]}\n\n"
                        f"Healthy reference script:\n{HEALTHY_TRANSFORM_SQL}"
                    )
                ),
            ]
        )
        if isinstance(result, FixPlan):
            return result
        if isinstance(result, dict):
            return FixPlan.model_validate(result)
    except Exception:  # noqa: BLE001 - deterministic fallback
        return None
    return None


# ---------------------------------------------------------------------------
# Graph assembly / entrypoint
# ---------------------------------------------------------------------------

_GRAPH: Any = None


def build_investigation_graph() -> Any:
    """Compile the three-node StateGraph."""
    graph = StateGraph(AgentState)
    graph.add_node("gather_evidence", gather_evidence)
    graph.add_node("analyze_and_rank", analyze_and_rank)
    graph.add_node("draft_safe_fix", draft_safe_fix)
    graph.add_edge(START, "gather_evidence")
    graph.add_edge("gather_evidence", "analyze_and_rank")
    graph.add_edge("analyze_and_rank", "draft_safe_fix")
    graph.add_edge("draft_safe_fix", END)
    return graph.compile()


def get_investigation_graph() -> Any:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_investigation_graph()
    return _GRAPH


def run_investigation(
    incident_id: int | None = None,
    table_name: str = "dim_customers",
    run_day: int = 15,
    job_name: str = JOB_NAME,
    file_path: str = TRANSFORM_SQL_PATH,
) -> dict[str, Any]:
    """Execute the investigation graph and return a validated report dict."""
    initial: AgentState = {
        "incident_id": incident_id,
        "table_name": table_name if table_name in SUPPORTED_TABLES else "dim_customers",
        "run_day": int(run_day),
        "job_name": job_name or JOB_NAME,
        "file_path": file_path or TRANSFORM_SQL_PATH,
        "mode": agent_mode(),
    }
    final_state = get_investigation_graph().invoke(initial)

    report = InvestigationReport(
        incident_id=incident_id,
        table_name=initial["table_name"],
        run_day=initial["run_day"],
        job_name=initial["job_name"],
        file_path=initial["file_path"],
        mode=str(final_state.get("mode") or agent_mode()),
        summary=str(final_state.get("summary") or "Investigation completed with no summary."),
        tool_trace=[
            ToolTraceEntry.model_validate(entry)
            for entry in (final_state.get("tool_trace") or [])
        ],
        ranked_causes=[
            RankedCause.model_validate(cause)
            for cause in (final_state.get("ranked_causes") or [])
        ],
        downstream_impact=list(final_state.get("downstream_impact") or []),
        proposed_sql=str(final_state.get("proposed_sql") or HEALTHY_TRANSFORM_SQL),
        rollback_plan=str(final_state.get("rollback_plan") or ""),
        generated_at=_now_iso(),
    )
    payload = report.model_dump()
    payload["project_root"] = str(PROJECT_ROOT)
    return payload
