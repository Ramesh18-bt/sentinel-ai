"""Typed contracts for the SentinelAI investigation agent.

The Pydantic models here are the wire format between the LangGraph agent, the
FastAPI backend, the SQLite ``incidents.investigation_json`` column and the
Streamlit forensic UI. Live LLM output is validated against the smaller
``CauseAnalysis`` / ``FixPlan`` models before being assembled into the final
``InvestigationReport``.
"""

from __future__ import annotations

import os
from typing import Any, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

AGENT_MODEL = os.getenv("SENTINEL_AGENT_MODEL", "gpt-4o-mini")
COPILOT_MODEL = os.getenv("SENTINEL_COPILOT_MODEL", "gpt-4o-mini")
TRANSFORM_SQL_PATH = "sql/transform_daily_customers.sql"


class ToolTraceEntry(BaseModel):
    """One forensic tool invocation captured for auditability."""

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    output: Any = None
    duration_ms: float | None = None


class RankedCause(BaseModel):
    """A ranked root-cause hypothesis with confidence and hard citations."""

    rank: int = 1
    cause: str
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class CauseAnalysis(BaseModel):
    """LLM-facing (or deterministic) analysis payload."""

    summary: str
    ranked_causes: list[RankedCause] = Field(default_factory=list)
    downstream_impact: list[str] = Field(default_factory=list)


class FixPlan(BaseModel):
    """LLM-facing remediation payload."""

    proposed_sql: str
    rollback_plan: str


class InvestigationReport(BaseModel):
    """Validated investigation stored on the incident record."""

    incident_id: int | None = None
    table_name: str
    run_day: int
    job_name: str
    file_path: str = TRANSFORM_SQL_PATH
    mode: str = "offline"
    summary: str
    tool_trace: list[ToolTraceEntry] = Field(default_factory=list)
    ranked_causes: list[RankedCause] = Field(default_factory=list)
    downstream_impact: list[str] = Field(default_factory=list)
    proposed_sql: str
    rollback_plan: str
    generated_at: str


class AgentState(TypedDict, total=False):
    """LangGraph state flowing through gather -> analyze -> draft."""

    incident_id: int | None
    table_name: str
    run_day: int
    job_name: str
    file_path: str
    mode: str
    rows_in: int
    rows_out: int
    tool_trace: list[dict[str, Any]]
    evidence: dict[str, Any]
    summary: str
    ranked_causes: list[dict[str, Any]]
    downstream_impact: list[str]
    anomaly_detected: bool
    proposed_sql: str
    rollback_plan: str
    error: str


def is_live_mode() -> bool:
    """Live mode requires both an explicit opt-out of mock mode and a key."""
    mock_flag = os.getenv("SENTINEL_MOCK_MODE", "true").strip().lower()
    has_key = bool(os.getenv("OPENAI_API_KEY", "").strip())
    return mock_flag in {"false", "0", "no", "off"} and has_key


def agent_mode() -> str:
    """Stable mode identifier used by the API/UI badges."""
    return "live" if is_live_mode() else "offline"


def agent_mode_label() -> str:
    """Human readable badge text for the executive header."""
    return "LIVE OPENAI AGENT" if is_live_mode() else "OFFLINE DETERMINISTIC ENGINE"
