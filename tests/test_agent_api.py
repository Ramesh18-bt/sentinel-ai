"""Agent, copilot, tools and API flow tests (need the project dependencies)."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langgraph")
fastapi = pytest.importorskip("fastapi")

from agent.chat_copilot import answer_question  # noqa: E402
from agent.graph import run_investigation  # noqa: E402
from api import main  # noqa: E402
from core.database import read_transform_sql  # noqa: E402
from core.tools import invoke_tool  # noqa: E402
from data.seed_generator import (  # noqa: E402
    inject_day15_anomaly,
    reset_and_seed_baseline,
    simulate_day15_healthy,
)


@pytest.fixture(autouse=True)
def fresh_baseline() -> None:
    reset_and_seed_baseline()


def test_tools_never_raise() -> None:
    out = invoke_tool("profile_upstream_table", {"table_name": "stg_users_cleaned", "run_day": 3})
    assert out["ok"] is True and out["row_count"] > 0


def test_readonly_sql_allows_harmless_and_blocks_writes() -> None:
    ok = invoke_tool(
        "run_readonly_diagnostic_sql",
        {"query": "SELECT replace(email, 'u', 'x') AS e, 'please update me' AS note FROM raw_users LIMIT 1"},
    )
    assert ok["ok"] is True
    for bad in (
        "DELETE FROM raw_users",
        "SELECT 1; DROP TABLE raw_users",
        "WITH x AS (SELECT 1) REPLACE INTO raw_users SELECT * FROM raw_users",
    ):
        assert invoke_tool("run_readonly_diagnostic_sql", {"query": bad})["ok"] is False


def test_healthy_day_investigation_blames_nobody() -> None:
    simulate_day15_healthy()
    report = run_investigation(run_day=15)
    assert "No data loss detected" in report["summary"]
    assert "NO ACTION REQUIRED" in report["rollback_plan"]
    assert report["downstream_impact"] == []


def test_anomaly_investigation_names_the_offending_commit() -> None:
    inject_day15_anomaly()
    report = run_investigation(run_day=15)
    top = report["ranked_causes"][0]
    assert "a8f92c1" in top["cause"] and "alex.dev" in top["cause"]
    assert all(entry["ok"] for entry in report["tool_trace"])


def test_approve_requires_investigation() -> None:
    incident_id = inject_day15_anomaly()["incident_id"]
    with pytest.raises(fastapi.HTTPException) as err:
        main.approve(incident_id)
    assert err.value.status_code == 409


def test_full_closed_loop_recovers() -> None:
    incident_id = inject_day15_anomaly()["incident_id"]
    main.investigate(incident_id)
    result = main.approve(incident_id)
    assert result["recovered"] is True
    assert result["rows_after"] == 101_200
    assert result["incident"]["status"] == "RESOLVED"
    assert "IS NOT NULL" not in read_transform_sql()
    assert main._system_state()["status"] == "RECOVERED"


def test_copilot_on_healthy_day_does_not_invent_a_root_cause() -> None:
    simulate_day15_healthy()
    reply = answer_question("Why did dim_customers drop on Day 15?")["reply"]
    assert "No drop to explain" in reply


def test_copilot_on_anomaly_cites_commit() -> None:
    inject_day15_anomaly()
    result = answer_question("Show me the exact git diff causing this")
    assert "a8f92c1" in result["reply"]
    assert json.dumps(result["tool_citations"])
