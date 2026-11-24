"""Read-only forensic tools for SentinelAI.

Every function is a LangChain ``@tool`` returning a JSON string. Nothing here
can mutate state: the only SQL that accepts user input is
``run_readonly_diagnostic_sql`` which is hard-blocked to single SELECT / WITH
statements.
"""

from __future__ import annotations

import functools
import json
import re
import sqlite3
from typing import Any, Callable

from langchain_core.tools import tool

from core.database import (
    JOB_NAME,
    NULL_COLUMNS,
    query_all,
    query_one,
    query_readonly,
)

MAX_ROWS = 200
FORBIDDEN_TOKENS = (
    "drop",
    "delete",
    "update",
    "insert",
    "alter",
    "create",
    "replace",
    "truncate",
    "attach",
    "detach",
    "pragma",
    "vacuum",
    "reindex",
    "grant",
    "revoke",
    "begin",
    "commit",
    "rollback",
)
_FORBIDDEN_RE = re.compile(r"\b(" + "|".join(FORBIDDEN_TOKENS) + r")\b", re.IGNORECASE)
# REPLACE is both a statement (blocked) and a harmless string function (allowed).
_REPLACE_FUNCTION_RE = re.compile(r"\breplace\s*\(", re.IGNORECASE)
_STRING_LITERAL_RE = re.compile(r"'(?:[^']|'')*'")
_QUOTED_IDENT_RE = re.compile(r'"(?:[^"]|"")*"')
_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)


def _strip_literals(sql_text: str) -> str:
    """Blank out comments, string literals and quoted identifiers.

    Keyword checks then only look at real SQL tokens, so ``SELECT 'please
    update me'`` is no longer rejected.
    """
    text = _COMMENT_RE.sub(" ", sql_text)
    text = _STRING_LITERAL_RE.sub("''", text)
    return _QUOTED_IDENT_RE.sub('""', text)


def _never_raise(tool_name: str) -> Callable[[Callable[..., str]], Callable[..., str]]:
    """Turn database errors into a JSON error payload instead of an exception.

    An exception inside a tool would abort the whole agent sweep; an error
    payload lets the agent (and the UI trace) see what went wrong and carry on.
    """

    def decorator(func: Callable[..., str]) -> Callable[..., str]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> str:
            try:
                return func(*args, **kwargs)
            except (sqlite3.Error, ValueError, TypeError) as exc:
                return _error(tool_name, f"{type(exc).__name__}: {exc}")

        return wrapper

    return decorator


def _dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, default=str, ensure_ascii=False)


def _error(tool_name: str, message: str, **extra: Any) -> str:
    payload = {"tool": tool_name, "ok": False, "error": message}
    payload.update(extra)
    return _dumps(payload)


# ---------------------------------------------------------------------------
# 1. Lineage
# ---------------------------------------------------------------------------


@tool
@_never_raise("inspect_lineage")
def inspect_lineage(table_name: str) -> str:
    """Inspect the data lineage graph around a table.

    Returns every upstream source feeding the table, the jobs that move the
    data, and the downstream assets (dashboards/marts) that break when this
    table is wrong. Read-only.
    """
    if not table_name or not table_name.strip():
        return _error("inspect_lineage", "table_name is required")

    table = table_name.strip()
    edges = query_all(
        "SELECT upstream_table, downstream_table, job_name FROM lineage_edges ORDER BY id"
    )
    if not edges:
        return _error("inspect_lineage", "lineage_edges is empty - run the seed generator")

    known = {table}
    upstream: list[dict[str, Any]] = []
    downstream: list[dict[str, Any]] = []
    seen_up: set[tuple[str, str, str]] = set()
    seen_down: set[tuple[str, str, str]] = set()

    # Walk upstream until closure.
    frontier = [table]
    while frontier:
        current = frontier.pop()
        for edge in edges:
            if edge["downstream_table"] != current:
                continue
            key = (edge["upstream_table"], edge["downstream_table"], edge["job_name"])
            if key in seen_up:
                continue
            seen_up.add(key)
            upstream.append(
                {
                    "upstream_table": edge["upstream_table"],
                    "downstream_table": edge["downstream_table"],
                    "job_name": edge["job_name"],
                }
            )
            if edge["upstream_table"] not in known:
                known.add(edge["upstream_table"])
                frontier.append(edge["upstream_table"])

    frontier = [table]
    while frontier:
        current = frontier.pop()
        for edge in edges:
            if edge["upstream_table"] != current:
                continue
            key = (edge["upstream_table"], edge["downstream_table"], edge["job_name"])
            if key in seen_down:
                continue
            seen_down.add(key)
            downstream.append(
                {
                    "upstream_table": edge["upstream_table"],
                    "downstream_table": edge["downstream_table"],
                    "job_name": edge["job_name"],
                }
            )
            if edge["downstream_table"] not in known:
                known.add(edge["downstream_table"])
                frontier.append(edge["downstream_table"])

    if not upstream and not downstream:
        return _error(
            "inspect_lineage",
            f"no lineage edges found for '{table}'",
            available_nodes=sorted({e["upstream_table"] for e in edges} | {e["downstream_table"] for e in edges}),
        )

    return _dumps(
        {
            "tool": "inspect_lineage",
            "ok": True,
            "table": table,
            "upstream_sources": upstream,
            "downstream_assets": downstream,
            "impacted_assets": [edge["downstream_table"] for edge in downstream],
            "impacted_jobs": [edge["job_name"] for edge in downstream],
            "total_edges": len(edges),
        }
    )


# ---------------------------------------------------------------------------
# 2. Upstream profiling
# ---------------------------------------------------------------------------


@tool
@_never_raise("profile_upstream_table")
def profile_upstream_table(table_name: str, run_day: int) -> str:
    """Profile an upstream table for one partition against its 14-day baseline.

    Reports the row count and the NULL percentage of the table's critical
    column (``registration_date`` for raw users), plus the rolling average so a
    spike is immediately visible. Read-only.
    """
    allowed = {"raw_users", "stg_users_cleaned", "dim_customers"}
    if table_name not in allowed:
        return _error("profile_upstream_table", f"table_name must be one of {sorted(allowed)}")

    null_column = NULL_COLUMNS.get(table_name, "email")
    row = query_one(
        f"SELECT COUNT(*) AS row_count FROM {table_name} WHERE run_day = ?", (int(run_day),)
    )
    row_count = int(row["row_count"]) if row else 0
    nulls = 0
    if row_count:
        null_row = query_one(
            f"SELECT COUNT(*) AS nulls FROM {table_name} WHERE run_day = ? AND {null_column} IS NULL",
            (int(run_day),),
        )
        nulls = int(null_row["nulls"]) if null_row else 0
    null_rate = round((nulls / row_count) * 100.0, 4) if row_count else 0.0

    history = query_all(
        """
        SELECT run_day, row_count, null_rate
        FROM metric_history
        WHERE table_name = ? AND run_day < ?
        ORDER BY run_day DESC
        LIMIT 14
        """,
        (table_name, int(run_day)),
    )
    if history:
        avg_rows = round(sum(float(item["row_count"]) for item in history) / len(history), 2)
        avg_null = round(sum(float(item["null_rate"]) for item in history) / len(history), 4)
        window = len(history)
    else:
        avg_rows = 0.0
        avg_null = 0.0
        window = 0

    return _dumps(
        {
            "tool": "profile_upstream_table",
            "ok": True,
            "table_name": table_name,
            "run_day": int(run_day),
            "null_column": null_column,
            "row_count": row_count,
            "null_count": nulls,
            "null_rate": null_rate,
            "baseline_14d_avg_row_count": avg_rows,
            "baseline_14d_avg_null_rate": avg_null,
            "baseline_window_days": window,
            "null_rate_delta_pp": round(null_rate - avg_null, 2),
            "row_count_delta_pct": (
                round(((row_count - avg_rows) / avg_rows) * 100.0, 2) if avg_rows else 0.0
            ),
        }
    )


# ---------------------------------------------------------------------------
# 3. Job logs
# ---------------------------------------------------------------------------


@tool
@_never_raise("query_job_logs")
def query_job_logs(job_name: str, run_day: int) -> str:
    """Fetch orchestration telemetry for a pipeline job and run day.

    Returns status, rows_in, rows_out, duration_sec and executed_at for every
    matching attempt, plus recent history for context. Read-only.
    """
    name = (job_name or JOB_NAME).strip()
    runs = query_all(
        """
        SELECT run_id, job_name, run_day, status, rows_in, rows_out, duration_sec, executed_at
        FROM pipeline_runs
        WHERE job_name = ? AND run_day = ?
        ORDER BY run_id DESC
        """,
        (name, int(run_day)),
    )
    recent = query_all(
        """
        SELECT run_id, job_name, run_day, status, rows_in, rows_out, duration_sec, executed_at
        FROM pipeline_runs
        WHERE job_name = ?
        ORDER BY run_id DESC
        LIMIT 5
        """,
        (name,),
    )
    if not runs:
        return _error(
            "query_job_logs",
            f"no runs logged for job '{name}' on run_day {run_day}",
            recent_runs=recent,
        )

    latest = runs[0]
    rows_in = int(latest["rows_in"] or 0)
    rows_out = int(latest["rows_out"] or 0)
    row_delta = rows_in - rows_out
    return _dumps(
        {
            "tool": "query_job_logs",
            "ok": True,
            "job_name": name,
            "run_day": int(run_day),
            "status": latest["status"],
            "rows_in": rows_in,
            "rows_out": rows_out,
            "dropped_rows": row_delta,
            "drop_pct": round((row_delta / rows_in) * 100.0, 2) if rows_in else 0.0,
            "duration_sec": float(latest["duration_sec"] or 0.0),
            "executed_at": latest["executed_at"],
            "attempts": runs,
            "recent_runs": recent,
            "silent_failure": bool(latest["status"] == "SUCCESS" and row_delta > 0),
        }
    )


# ---------------------------------------------------------------------------
# 4. Git forensics
# ---------------------------------------------------------------------------


@tool
@_never_raise("get_git_diff")
def get_git_diff(file_path: str) -> str:
    """Return recent commits (and their diffs) touching a file path.

    This is the change-correlation tool used to pin a data incident on a code
    change. Read-only.
    """
    path = (file_path or "").strip()
    if not path:
        return _error("get_git_diff", "file_path is required")

    commits = query_all(
        """
        SELECT commit_hash, author, message, file_path, diff_text, committed_at
        FROM code_commits
        WHERE file_path = ? OR file_path LIKE ?
        ORDER BY committed_at DESC, commit_hash DESC
        """,
        (path, f"%{path.split('/')[-1]}%"),
    )
    if not commits:
        commits = query_all(
            """
            SELECT commit_hash, author, message, file_path, diff_text, committed_at
            FROM code_commits
            ORDER BY committed_at DESC, commit_hash DESC
            LIMIT 5
            """
        )
        if not commits:
            return _error("get_git_diff", "code_commits is empty")

    latest = commits[0]
    # Commits whose diff adds a row-filtering NULL predicate are the ones that
    # can silently drop records; the analysis should only blame those.
    suspects = [
        commit
        for commit in commits
        if re.search(r"^\+.*\bIS\s+NOT\s+NULL\b", str(commit.get("diff_text") or ""), re.IGNORECASE | re.MULTILINE)
    ]
    return _dumps(
        {
            "tool": "get_git_diff",
            "ok": True,
            "file_path": path,
            "relative_path": path,
            "commit_count": len(commits),
            "latest_commit": latest["commit_hash"],
            "latest_author": latest["author"],
            "latest_message": latest["message"],
            "latest_committed_at": latest["committed_at"],
            "diff_text": latest["diff_text"],
            "suspect_commits": suspects,
            "commits": commits,
        }
    )


# ---------------------------------------------------------------------------
# 5. Ad-hoc read-only SQL
# ---------------------------------------------------------------------------


@tool
@_never_raise("run_readonly_diagnostic_sql")
def run_readonly_diagnostic_sql(query: str) -> str:
    """Execute a strictly read-only SELECT/WITH diagnostic query.

    Blocks any DDL/DML keyword (DROP, DELETE, UPDATE, INSERT, ALTER, ...),
    rejects stacked statements, and caps results at 200 rows. Read-only.
    """
    if not query or not query.strip():
        return _error("run_readonly_diagnostic_sql", "query is required")

    cleaned = query.strip().rstrip(";").strip()
    tokens = _strip_literals(cleaned)
    lowered = tokens.strip().lower()

    if not (lowered.startswith("select") or lowered.startswith("with")):
        return _error(
            "run_readonly_diagnostic_sql",
            "only SELECT or WITH queries are permitted",
        )
    if ";" in tokens:
        return _error("run_readonly_diagnostic_sql", "stacked statements are not permitted")
    forbidden = _FORBIDDEN_RE.search(_REPLACE_FUNCTION_RE.sub("fn_replace(", tokens))
    if forbidden:
        return _error(
            "run_readonly_diagnostic_sql",
            f"forbidden keyword detected: {forbidden.group(0).upper()}",
        )

    try:
        # Read-only connection: even if a write slipped past the filter,
        # SQLite would refuse it.
        rows = query_readonly(cleaned)
    except Exception as exc:  # noqa: BLE001 - surfaced to the agent as JSON
        return _error("run_readonly_diagnostic_sql", f"query failed: {exc}", query=cleaned)

    truncated = len(rows) > MAX_ROWS
    rows = rows[:MAX_ROWS]
    columns = list(rows[0].keys()) if rows else []
    return _dumps(
        {
            "tool": "run_readonly_diagnostic_sql",
            "ok": True,
            "query": cleaned,
            "columns": columns,
            "row_count": len(rows),
            "truncated": truncated,
            "rows": rows,
        }
    )


FORENSIC_TOOLS = [
    inspect_lineage,
    profile_upstream_table,
    query_job_logs,
    get_git_diff,
    run_readonly_diagnostic_sql,
]

TOOL_REGISTRY = {item.name: item for item in FORENSIC_TOOLS}


def invoke_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Invoke a forensic tool by name and return parsed JSON for traces."""
    tool_obj = TOOL_REGISTRY.get(tool_name)
    if tool_obj is None:
        raise KeyError(f"unknown forensic tool: {tool_name}")
    raw = tool_obj.invoke(arguments)
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"tool": tool_name, "ok": False, "error": raw}
    return raw  # type: ignore[return-value]
