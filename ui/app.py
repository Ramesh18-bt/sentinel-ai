"""SentinelAI Executive Command Center (Streamlit).

Three surfaces on top of the FastAPI backend:
  1. Observability & lineage telemetry
  2. Autonomous root cause + human-in-the-loop approval
  3. SentinelAI Data Copilot chat

Run with:  uv run streamlit run ui/app.py
"""

from __future__ import annotations

import json
import os
from html import escape
from typing import Any

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("SENTINEL_API_URL", "http://localhost:8000").rstrip("/")
QUICK_PROMPTS = [
    "Why did dim_customers drop on Day 15?",
    "Which dashboards are impacted?",
    "Show me the exact git diff causing this",
    "Compare Day 15 null rate vs 14-day baseline",
    "What job telemetry did the pipeline emit?",
    "What is the safe fix and rollback plan?",
]

st.set_page_config(
    page_title="SentinelAI · Data Reliability Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg: #05080f;
    --panel: #0b1220;
    --panel-2: #0f172a;
    --border: #1e293b;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --emerald: #10b981;
    --crimson: #ef4444;
    --cyan: #22d3ee;
    --amber: #f59e0b;
    --sky: #38bdf8;
}

html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.stApp {
    background:
        radial-gradient(1200px 600px at 12% -10%, rgba(56,189,248,0.10), transparent 60%),
        radial-gradient(900px 500px at 92% 0%, rgba(16,185,129,0.07), transparent 55%),
        var(--bg);
    color: var(--text);
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070c16 0%, #0a101c 100%);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text); }
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1500px; }
h1, h2, h3, h4 { letter-spacing: -0.02em; color: #f8fafc; }

.hero {
    display: flex; align-items: center; justify-content: space-between; gap: 1.2rem;
    padding: 1.4rem 1.7rem; margin-bottom: 1.1rem;
    background: linear-gradient(120deg, rgba(15,23,42,0.95), rgba(8,15,28,0.98));
    border: 1px solid var(--border);
    border-radius: 18px;
    box-shadow: 0 24px 60px -32px rgba(0,0,0,0.95), inset 0 1px 0 rgba(255,255,255,0.04);
}
.hero-left { display: flex; flex-direction: column; gap: 0.35rem; }
.hero-eyebrow { font-size: 0.68rem; letter-spacing: 0.32em; text-transform: uppercase; color: var(--sky); font-weight: 600; }
.hero-title { font-size: 1.85rem; font-weight: 800; margin: 0; color: #f8fafc; }
.hero-title span { color: var(--emerald); }
.hero-sub { color: var(--muted); font-size: 0.86rem; max-width: 62ch; line-height: 1.5; }
.hero-right { display: flex; flex-direction: column; align-items: flex-end; gap: 0.55rem; }

.pill {
    display: inline-flex; align-items: center; gap: 0.5rem;
    padding: 0.45rem 0.95rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 800; letter-spacing: 0.14em; text-transform: uppercase;
    border: 1px solid transparent; white-space: nowrap;
}
.pill .dot { width: 8px; height: 8px; border-radius: 50%; }
.pill-healthy { color: #6ee7b7; background: rgba(16,185,129,0.12); border-color: rgba(16,185,129,0.45); }
.pill-healthy .dot { background: var(--emerald); box-shadow: 0 0 10px var(--emerald); }
.pill-critical { color: #fecaca; background: rgba(239,68,68,0.14); border-color: rgba(239,68,68,0.55); animation: pulse 1.8s infinite; }
.pill-critical .dot { background: var(--crimson); box-shadow: 0 0 12px var(--crimson); }
.pill-recovered { color: #a5f3fc; background: rgba(34,211,238,0.12); border-color: rgba(34,211,238,0.5); }
.pill-recovered .dot { background: var(--cyan); box-shadow: 0 0 10px var(--cyan); }
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(239,68,68,0.55); }
    70% { box-shadow: 0 0 0 12px rgba(239,68,68,0); }
    100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
}

.mode-badge {
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; letter-spacing: 0.1em;
    color: #cbd5e1; background: rgba(148,163,184,0.10);
    border: 1px dashed rgba(148,163,184,0.45);
    padding: 0.35rem 0.7rem; border-radius: 8px; text-transform: uppercase;
}

.kpi-card {
    background: linear-gradient(160deg, rgba(15,23,42,0.96), rgba(8,13,24,0.98));
    border: 1px solid var(--border); border-radius: 14px;
    padding: 1rem 1.15rem 0.9rem; height: 100%;
    border-top: 3px solid var(--sky);
    box-shadow: 0 18px 40px -34px rgba(0,0,0,0.9);
}
.kpi-card.emerald { border-top-color: var(--emerald); }
.kpi-card.crimson { border-top-color: var(--crimson); }
.kpi-card.amber { border-top-color: var(--amber); }
.kpi-label { font-size: 0.66rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--muted); font-weight: 700; min-height: 2.2em; display: flex; align-items: flex-end; }
.kpi-value { font-size: 1.85rem; font-weight: 800; color: #f8fafc; margin: 0.3rem 0 0.15rem; line-height: 1.1; }
.kpi-delta { font-size: 0.78rem; font-weight: 700; }
.kpi-delta.up { color: #34d399; }
.kpi-delta.down { color: #f87171; }
.kpi-delta.flat { color: var(--muted); }
.kpi-sub { font-size: 0.7rem; color: #64748b; margin-top: 0.35rem; letter-spacing: 0.03em; }

.section-title {
    font-size: 0.72rem; letter-spacing: 0.24em; text-transform: uppercase;
    color: var(--sky); font-weight: 700; margin: 0.6rem 0 0.4rem;
}
.panel {
    background: rgba(11,18,32,0.85); border: 1px solid var(--border);
    border-radius: 16px; padding: 1rem 1.2rem;
}

.incident-card {
    border: 1px solid rgba(239,68,68,0.45);
    background: linear-gradient(130deg, rgba(60,10,16,0.85), rgba(11,18,32,0.95));
    border-radius: 16px; padding: 1.1rem 1.3rem; margin-bottom: 0.9rem;
}
.incident-head { display: flex; align-items: center; gap: 0.7rem; flex-wrap: wrap; }
.sev-tag {
    font-size: 0.64rem; font-weight: 800; letter-spacing: 0.18em; text-transform: uppercase;
    padding: 0.25rem 0.6rem; border-radius: 6px;
}
.sev-critical { background: rgba(239,68,68,0.2); color: #fca5a5; border: 1px solid rgba(239,68,68,0.6); }
.sev-warning { background: rgba(245,158,11,0.18); color: #fcd34d; border: 1px solid rgba(245,158,11,0.55); }
.sev-resolved { background: rgba(16,185,129,0.18); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.55); }
.incident-title { font-size: 1.02rem; font-weight: 700; color: #f8fafc; }
.incident-summary { color: #cbd5e1; font-size: 0.87rem; line-height: 1.6; margin-top: 0.65rem; }
.incident-meta { color: #64748b; font-size: 0.72rem; margin-top: 0.6rem; font-family: 'JetBrains Mono', monospace; }

.cause-card {
    background: linear-gradient(160deg, rgba(15,23,42,0.95), rgba(9,14,26,0.98));
    border: 1px solid var(--border); border-left: 3px solid var(--sky);
    border-radius: 12px; padding: 0.9rem 1.1rem; margin-bottom: 0.75rem;
}
.cause-card.c1 { border-left-color: var(--crimson); }
.cause-card.c2 { border-left-color: var(--amber); }
.cause-card.c3 { border-left-color: var(--sky); }
.cause-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem; }
.cause-rank { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #94a3b8; letter-spacing: 0.1em; }
.cause-conf { font-size: 0.75rem; font-weight: 800; color: #f8fafc; }
.cause-text { color: #e2e8f0; font-size: 0.88rem; line-height: 1.55; }
.bar { height: 6px; border-radius: 999px; background: rgba(148,163,184,0.16); margin: 0.65rem 0 0.55rem; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--sky), var(--emerald)); }
.cite-row { display: flex; flex-wrap: wrap; gap: 0.35rem; }
.cite-chip {
    font-family: 'JetBrains Mono', monospace; font-size: 0.66rem; color: #bae6fd;
    background: rgba(56,189,248,0.10); border: 1px solid rgba(56,189,248,0.28);
    border-radius: 999px; padding: 0.16rem 0.55rem; white-space: nowrap;
}

.scorecard { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 0.5rem; }
.score-item {
    flex: 1 1 160px; background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.35); border-radius: 12px; padding: 0.8rem 1rem;
}
.score-item .label { font-size: 0.65rem; letter-spacing: 0.18em; text-transform: uppercase; color: #6ee7b7; font-weight: 700; }
.score-item .value { font-size: 1.35rem; font-weight: 800; color: #f8fafc; margin-top: 0.15rem; }
.score-item .note { font-size: 0.7rem; color: #94a3b8; }

.rollback-box {
    background: rgba(245,158,11,0.07); border: 1px solid rgba(245,158,11,0.35);
    border-radius: 12px; padding: 1rem 1.2rem; color: #fde68a;
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; line-height: 1.75;
    white-space: pre-wrap;
}
.tool-pill {
    display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: 0.66rem;
    padding: 0.18rem 0.6rem; border-radius: 999px; margin: 0 0.3rem 0.3rem 0;
    background: rgba(148,163,184,0.12); border: 1px solid rgba(148,163,184,0.3); color: #cbd5e1;
}
.tool-pill.ok { color: #6ee7b7; border-color: rgba(16,185,129,0.45); background: rgba(16,185,129,0.10); }
.tool-pill.err { color: #fca5a5; border-color: rgba(239,68,68,0.45); background: rgba(239,68,68,0.10); }

div.stButton > button {
    border-radius: 10px; border: 1px solid var(--border); font-weight: 600;
    background: rgba(15,23,42,0.9); color: #e2e8f0; transition: all 0.15s ease;
}
div.stButton > button:hover { border-color: var(--sky); color: #f8fafc; box-shadow: 0 0 0 1px rgba(56,189,248,0.35); }
div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #059669, #10b981); border: none; color: #04120c;
    font-weight: 800; letter-spacing: 0.02em;
}
div.stButton > button[kind="primary"]:hover { filter: brightness(1.08); box-shadow: 0 10px 30px -12px rgba(16,185,129,0.7); }
.stTabs [data-baseweb="tab-list"] { gap: 0.4rem; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] {
    background: transparent; border-radius: 10px 10px 0 0; padding: 0.55rem 1.1rem;
    color: #94a3b8; font-weight: 600;
}
.stTabs [aria-selected="true"] { color: #f8fafc !important; background: rgba(56,189,248,0.08); border-bottom: 2px solid var(--sky); }
[data-testid="stExpander"] {
    background: rgba(11,18,32,0.85); border: 1px solid var(--border); border-radius: 12px;
}
[data-testid="stChatMessage"] {
    background: rgba(11,18,32,0.85); border: 1px solid var(--border); border-radius: 14px;
    padding: 0.5rem 0.75rem; margin-bottom: 0.55rem;
}
code, pre, .stCode { font-family: 'JetBrains Mono', monospace !important; }
div[data-testid="stMetricValue"] { color: #f8fafc; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def api_get(path: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = httpx.get(f"{API_URL}{path}", timeout=90.0)
        response.raise_for_status()
        return response.json(), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def api_post(path: str, payload: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = httpx.post(f"{API_URL}{path}", json=payload or {}, timeout=180.0)
        if response.status_code >= 400:
            detail = response.text[:400]
            try:
                detail = response.json().get("detail", detail)
            except Exception:  # noqa: BLE001
                pass
            return None, f"HTTP {response.status_code}: {detail}"
        return response.json(), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def fmt(number: Any) -> str:
    try:
        return f"{int(float(number)):,}"
    except (TypeError, ValueError):
        return "—"


def pct(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value):+.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def telemetry_figure(metric_history: list[dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    frame = pd.DataFrame(metric_history)
    if frame.empty:
        fig.update_layout(template="plotly_dark", height=420, title="No telemetry yet")
        return fig

    dim = frame[frame["table_name"] == "dim_customers"].sort_values("run_day")
    raw = frame[frame["table_name"] == "raw_users"].sort_values("run_day")
    if dim.empty:
        fig.update_layout(template="plotly_dark", height=420, title="No dim_customers telemetry")
        return fig

    baseline = dim[dim["run_day"] <= 14]
    mean = float(baseline["row_count"].mean()) if not baseline.empty else float(dim["row_count"].mean())
    std = float(baseline["row_count"].std(ddof=0)) if len(baseline) > 1 else 0.0
    upper = mean + 1.96 * std
    lower = mean - 1.96 * std
    days = dim["run_day"].astype(int).tolist()

    fig.add_trace(
        go.Scatter(
            x=days, y=[upper] * len(days), mode="lines", line=dict(width=0),
            hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days, y=[lower] * len(days), mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(56,189,248,0.09)",
            name="Normal band (95%)", hoverinfo="skip",
        )
    )
    fig.add_hline(
        y=mean, line=dict(color="#38bdf8", width=1, dash="dot"),
        annotation_text=f"14-day mean {mean:,.0f}",
        annotation_position="top right",
        annotation_font=dict(color="#38bdf8", size=11),
    )

    if not raw.empty:
        fig.add_trace(
            go.Scatter(
                x=raw["run_day"], y=raw["row_count"], mode="lines+markers",
                name="raw_users (ingested)",
                line=dict(color="#64748b", width=2),
                marker=dict(size=5, color="#64748b"),
                hovertemplate="Day %{x}<br>raw_users %{y:,}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=dim["run_day"], y=dim["row_count"], mode="lines+markers",
            name="dim_customers (materialised)",
            line=dict(color="#10b981", width=3),
            marker=dict(size=7, color="#10b981", line=dict(color="#022c22", width=1)),
            hovertemplate="Day %{x}<br>dim_customers %{y:,}<extra></extra>",
        )
    )

    drop_threshold = mean * 0.85
    anomalies = dim[dim["row_count"] < drop_threshold]
    for _, row in anomalies.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["run_day"]], y=[row["row_count"]], mode="markers+text",
                marker=dict(size=15, color="#ef4444", symbol="x", line=dict(width=2, color="#7f1d1d")),
                text=[f"  -{abs((row['row_count'] - mean) / mean * 100):.1f}%"],
                textposition="middle right",
                textfont=dict(color="#fca5a5", size=12),
                name=f"Day {int(row['run_day'])} anomaly",
                hovertemplate=f"Day {int(row['run_day'])}<br>rows %{{y:,}}<br>ANOMALY<extra></extra>",
                showlegend=False,
            )
        )
        fig.add_annotation(
            x=row["run_day"], y=row["row_count"],
            text=f"Day {int(row['run_day'])} drop<br>{fmt(row['row_count'])} rows",
            showarrow=True, arrowhead=2, arrowcolor="#ef4444",
            ax=-70, ay=-55,
            font=dict(color="#fecaca", size=11),
            bgcolor="rgba(69,10,10,0.85)", bordercolor="#ef4444", borderwidth=1,
        )

    if not anomalies.empty:
        last = dim.sort_values("run_day").iloc[-1]
        if int(last["run_day"]) > int(anomalies["run_day"].max()) and last["row_count"] >= mean * 0.95:
            fig.add_trace(
                go.Scatter(
                    x=[last["run_day"]], y=[last["row_count"]], mode="markers+text",
                    marker=dict(size=16, color="#22d3ee", symbol="star"),
                    text=["  RECOVERED"], textposition="middle right",
                    textfont=dict(color="#a5f3fc", size=12),
                    name="Recovery confirmed", showlegend=False,
                    hovertemplate="Day %{x}<br>recovered %{y:,}<extra></extra>",
                )
            )

    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=10, r=20, t=58, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,8,15,0.6)",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(
            title="run_day", gridcolor="rgba(148,163,184,0.10)", dtick=1,
            range=[0.5, float(max(days)) + 0.5],
        ),
        yaxis=dict(title="rows", gridcolor="rgba(148,163,184,0.10)", tickformat=","),
        font=dict(family="Inter, sans-serif", color="#cbd5e1"),
    )
    return fig


def lineage_figure(
    edges: list[dict[str, Any]],
    system: dict[str, Any],
    incidents: list[dict[str, Any]],
) -> go.Figure:
    if not edges:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", height=380, title="No lineage edges")
        return fig

    nodes: list[str] = []
    for edge in edges:
        for key in ("upstream_table", "downstream_table"):
            if edge[key] not in nodes:
                nodes.append(edge[key])
    index = {name: idx for idx, name in enumerate(nodes)}

    active = [row for row in incidents if row.get("status") != "RESOLVED"]
    impacted: set[str] = set()
    if active:
        incident_table = str(active[0]["table_name"])
        impacted.add(incident_table)
        frontier = [incident_table]
        while frontier:
            current = frontier.pop()
            for edge in edges:
                if edge["upstream_table"] == current and edge["downstream_table"] not in impacted:
                    impacted.add(str(edge["downstream_table"]))
                    frontier.append(str(edge["downstream_table"]))

    upstream_only = {
        str(edge["upstream_table"])
        for edge in edges
        if str(edge["downstream_table"]) in impacted and str(edge["upstream_table"]) not in impacted
    }

    node_colors = []
    for name in nodes:
        if name in impacted:
            node_colors.append("#ef4444")
        elif name in upstream_only:
            node_colors.append("#f59e0b")
        else:
            node_colors.append("#38bdf8")

    raw_rows = max(int(system.get("raw_rows") or 1), 1)
    dim_rows = max(int(system.get("dim_rows") or 1), 1)
    link_values = []
    for edge in edges:
        if edge["upstream_table"] == "raw_users":
            link_values.append(raw_rows)
        elif edge["downstream_table"] == "exec_revenue_dashboard":
            link_values.append(max(dim_rows, 1))
        else:
            link_values.append(max(dim_rows, 1))

    link_colors = [
        "rgba(239,68,68,0.35)" if str(edge["downstream_table"]) in impacted
        else "rgba(56,189,248,0.22)"
        for edge in edges
    ]

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                label=[f"{name}  ({fmt(raw_rows) if name == 'raw_users' else fmt(dim_rows)})" for name in nodes],
                pad=24,
                thickness=18,
                color=node_colors,
                line=dict(color="rgba(226,232,240,0.35)", width=1),
                hovertemplate="%{label}<extra></extra>",
            ),
            link=dict(
                source=[index[str(edge["upstream_table"])] for edge in edges],
                target=[index[str(edge["downstream_table"])] for edge in edges],
                value=link_values,
                color=link_colors,
                label=[str(edge["job_name"]) for edge in edges],
                hovertemplate="%{label}<br>%{value:,} rows<extra></extra>",
            ),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=340,
        margin=dict(l=28, r=28, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0", size=12),
    )
    return fig


# ---------------------------------------------------------------------------
# HTML fragments
# ---------------------------------------------------------------------------


def kpi_card(label: str, value: str, delta: str, delta_class: str, sub: str, accent: str) -> str:
    return f"""
    <div class="kpi-card {accent}">
        <div class="kpi-label">{escape(label)}</div>
        <div class="kpi-value">{escape(value)}</div>
        <div class="kpi-delta {delta_class}">{escape(delta)}</div>
        <div class="kpi-sub">{escape(sub)}</div>
    </div>
    """


def cause_card(cause: dict[str, Any]) -> str:
    rank = int(cause.get("rank") or 1)
    confidence = float(cause.get("confidence") or 0.0)
    citations = "".join(
        f'<span class="cite-chip">{escape(str(citation))}</span>'
        for citation in (cause.get("citations") or [])
    )
    return f"""
    <div class="cause-card c{min(rank, 3)}">
        <div class="cause-head">
            <span class="cause-rank">CAUSE #{rank}</span>
            <span class="cause-conf">{confidence * 100:.0f}% confidence</span>
        </div>
        <div class="cause-text">{escape(str(cause.get("cause", "")))}</div>
        <div class="bar"><div class="bar-fill" style="width:{max(min(confidence * 100, 100), 2):.0f}%"></div></div>
        <div class="cite-row">{citations}</div>
    </div>
    """


def tool_pills(citations: list[dict[str, Any]]) -> str:
    pills = []
    for item in citations:
        ok = bool(item.get("ok", False))
        css = "ok" if ok else "err"
        arguments = item.get("arguments") or {}
        args_text = ", ".join(f"{key}={value}" for key, value in arguments.items())
        label = str(item.get("tool", "tool"))
        pills.append(f'<span class="tool-pill {css}">{escape(label)}({escape(args_text[:60])})</span>')
    return "".join(pills) or '<span class="tool-pill">no tools invoked</span>'


# ---------------------------------------------------------------------------
# Sidebar control center
# ---------------------------------------------------------------------------

flash = st.session_state.pop("flash", None)
if flash:
    # Flash entries are (message, ok); plain strings are treated as success.
    flash_message, flash_ok = flash if isinstance(flash, tuple) else (flash, True)
    st.toast(flash_message, icon="✅" if flash_ok else "⚠️")

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.2rem;">
            <div style="font-size:1.6rem;">🛡️</div>
            <div>
                <div style="font-size:1.05rem;font-weight:800;color:#f8fafc;letter-spacing:-0.02em;">SentinelAI</div>
                <div style="font-size:0.65rem;letter-spacing:0.22em;text-transform:uppercase;color:#64748b;">Control Center</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown('<div class="section-title">Simulation Workflow</div>', unsafe_allow_html=True)

    if st.button("1. Reset Baseline (Days 1–14)", width="stretch"):
        payload, error = api_post("/reset")
        st.session_state["flash"] = (
            (f"Baseline reseeded · {fmt(payload.get('raw_users_rows'))} raw rows across 14 days", True)
            if payload else (f"Reset failed: {error}", False)
        )
        st.rerun()

    if st.button("2. Run Healthy Day 15", width="stretch"):
        payload, error = api_post("/simulate/healthy")
        st.session_state["flash"] = (
            (f"Healthy Day 15 landed · {fmt(payload.get('dim_customers_rows'))} rows · quality {payload.get('quality_status')}", True)
            if payload else (f"Healthy simulation failed: {error}", False)
        )
        st.rerun()

    if st.button("3. Inject Day 15 Anomaly", width="stretch", type="primary"):
        payload, error = api_post("/simulate/anomaly")
        st.session_state["flash"] = (
            (
                f"Anomaly injected · dim_customers {fmt(payload.get('dim_customers_rows'))} rows "
                f"({payload.get('drop_pct_vs_baseline')}%) · incident #{payload.get('incident_id')}",
                True,
            )
            if payload else (f"Anomaly injection failed: {error}", False)
        )
        st.rerun()

    st.divider()
    st.markdown('<div class="section-title">Active SQL File</div>', unsafe_allow_html=True)
    st.caption("sql/transform_daily_customers.sql")
    sql_slot = st.empty()
    st.divider()
    st.markdown(
        f'<div class="kpi-sub">Backend<br><span style="color:#38bdf8;font-family:JetBrains Mono,monospace;">{escape(API_URL)}</span></div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Data load
# ---------------------------------------------------------------------------

data, error = api_get("/dashboard-data")
if error or not data:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-left">
                <div class="hero-eyebrow">SentinelAI</div>
                <div class="hero-title">Backend <span style="color:#ef4444;">offline</span></div>
                <div class="hero-sub">The dashboard could not reach the FastAPI backend.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code("uv run uvicorn api.main:app --port 8000", language="bash")
    st.error(f"API error: {error}")
    st.stop()

system = data.get("system", {})
incidents = data.get("incidents", []) or []
raw_rows = int(system.get("raw_rows") or 0)
dim_rows = int(system.get("dim_rows") or 0)
baseline_rows = float(system.get("baseline_rows") or 0.0)
variance_pct = float(system.get("variance_pct") or 0.0)
run_day = int(system.get("run_day") or 15)
status = str(system.get("status") or "HEALTHY")
mode_label = str(system.get("agent_mode_label") or "OFFLINE DETERMINISTIC ENGINE")
transform_sql = str(system.get("transform_sql") or "")

status_class = {
    "HEALTHY": "pill-healthy",
    "CRITICAL INCIDENT": "pill-critical",
    "RECOVERED": "pill-recovered",
}.get(status, "pill-healthy")

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-left">
            <div class="hero-eyebrow">Autonomous Data Reliability</div>
            <div class="hero-title">Sentinel<span>AI</span> Command Center</div>
            <div class="hero-sub">
                Continuous quality guardrails, autonomous root-cause investigation and a
                closed-loop fix approval gate across the customer revenue pipeline.
            </div>
        </div>
        <div class="hero-right">
            <span class="pill {status_class}"><span class="dot"></span>{escape(status)}</span>
            <span class="mode-badge">{escape(mode_label)}</span>
            <span class="kpi-sub" style="margin:0;">run_day {run_day} · {len(incidents)} incident(s) tracked</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

sql_slot.code(transform_sql or "-- transform SQL unavailable", language="sql")

# KPI row -------------------------------------------------------------------
variance_class = "down" if variance_pct < -2 else ("up" if variance_pct > 2 else "flat")
sla = str(system.get("downstream_sla") or "ON TRACK")
sla_class = "down" if sla != "ON TRACK" else "up"
cards = [
    ("RAW USERS VOLUME", fmt(raw_rows), f"{fmt(raw_rows - baseline_rows)} vs baseline", "flat", f"run_day {run_day} ingested", ""),
    ("DIM CUSTOMERS VOLUME", fmt(dim_rows), f"{fmt(dim_rows - baseline_rows)} vs baseline", variance_class, f"14-day baseline {fmt(baseline_rows)}", "emerald" if variance_pct > -5 else "crimson"),
    ("BASELINE VARIANCE", pct(variance_pct), "14-day rolling mean", variance_class, "guardrail trips at -15.0%", "amber" if -15 < variance_pct < 0 else ("crimson" if variance_pct <= -15 else "")),
    ("DOWNSTREAM SLA", sla, "exec_revenue_dashboard", sla_class, "build_exec_mart consumer", "emerald" if sla == "ON TRACK" else "crimson"),
]
columns = st.columns(4, gap="small")
for column, (label, value, delta, delta_class, sub, accent) in zip(columns, cards, strict=True):
    with column:
        st.markdown(kpi_card(label, value, delta, delta_class, sub, accent), unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_obs, tab_rca, tab_chat = st.tabs(
    ["📡 Observability & Lineage", "🧠 Autonomous Root Cause & HITL", "💬 SentinelAI Data Copilot"]
)

# ------------------------------------------------------------------ Tab 1
with tab_obs:
    st.markdown('<div class="section-title">Dual-Series Telemetry · Days 1–15</div>', unsafe_allow_html=True)
    st.plotly_chart(
        telemetry_figure(data.get("metric_history", [])),
        width="stretch",
        config={"displayModeBar": False},
    )
    st.caption(
        "raw_users ingested vs dim_customers materialised · shaded band = 95% normal range "
        "around the 14-day baseline · red X marks the Day 15 anomaly."
    )

    st.markdown('<div class="section-title">Pipeline Lineage Graph</div>', unsafe_allow_html=True)
    st.plotly_chart(
        lineage_figure(data.get("lineage_edges", []), system, incidents),
        width="stretch",
        config={"displayModeBar": False},
    )
    st.caption(
        "Nodes turn crimson when an active incident contaminates them and amber when they "
        "feed an impacted consumer."
    )

    st.markdown('<div class="section-title">Pipeline Run Ledger</div>', unsafe_allow_html=True)
    runs = pd.DataFrame(data.get("pipeline_runs", []))
    if runs.empty:
        st.info("No pipeline runs recorded yet.")
    else:
        runs = runs.sort_values(["run_day", "run_id"], ascending=[False, False]).head(18)
        runs["row_delta"] = runs["rows_in"].fillna(0).astype(int) - runs["rows_out"].fillna(0).astype(int)
        st.dataframe(
            runs[
                ["run_id", "job_name", "run_day", "status", "rows_in", "rows_out", "row_delta", "duration_sec", "executed_at"]
            ],
            width="stretch",
            hide_index=True,
        )

    st.markdown('<div class="section-title">Change Feed</div>', unsafe_allow_html=True)
    commits = data.get("code_commits", []) or []
    for commit in commits[:4]:
        suspicious = "registration_date IS NOT NULL" in str(commit.get("diff_text", ""))
        tag = '<span class="sev-tag sev-critical">SUSPECT</span>' if suspicious else '<span class="sev-tag sev-resolved">CLEAN</span>'
        st.markdown(
            f"""
            <div class="panel" style="margin-bottom:0.5rem;padding:0.75rem 1rem;">
                <div class="incident-head">
                    <span class="cause-rank">{escape(str(commit.get('commit_hash')))}</span>
                    {tag}
                    <span style="color:#e2e8f0;font-weight:600;font-size:0.86rem;">{escape(str(commit.get('message')))}</span>
                </div>
                <div class="incident-meta">{escape(str(commit.get('author')))} · {escape(str(commit.get('file_path')))} · {escape(str(commit.get('committed_at')))}</div>
                <div class="incident-meta" style="color:#fca5a5;">{escape(str(commit.get('diff_text')))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ------------------------------------------------------------------ Tab 2
with tab_rca:
    if not incidents:
        st.markdown(
            """
            <div class="panel">
                <div class="section-title">No active incident</div>
                <div class="incident-summary">
                    The quality engine is idle. Use <b>3. Inject Day 15 Anomaly</b> in the sidebar to
                    simulate the silent -37.6% customer-dimension drop, then launch the autonomous
                    investigation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        incident = incidents[0]
        investigation = None
        if incident.get("investigation_json"):
            try:
                investigation = json.loads(incident["investigation_json"])
            except json.JSONDecodeError:
                investigation = None

        sev = str(incident.get("severity") or "CRITICAL")
        inc_status = str(incident.get("status") or "OPEN")
        tag_class = "sev-resolved" if inc_status == "RESOLVED" else ("sev-warning" if sev == "WARNING" else "sev-critical")
        incident_meta = [
            f"anomaly_type={escape(str(incident.get('anomaly_type')))}",
            f"created={escape(str(incident.get('created_at')))}",
        ]
        if incident.get("approved_by"):
            incident_meta.append(f"approved_by={escape(str(incident.get('approved_by')))}")
        if incident.get("resolved_at"):
            incident_meta.append(f"resolved_at={escape(str(incident.get('resolved_at')))}")
        incident_meta_line = " · ".join(incident_meta)
        st.markdown(
            f"""
            <div class="incident-card">
                <div class="incident-head">
                    <span class="sev-tag {tag_class}">{escape(sev)} · {escape(inc_status)}</span>
                    <span class="incident-title">Incident #{incident.get('incident_id')} · {escape(str(incident.get('table_name')))} · run_day {incident.get('run_day')}</span>
                </div>
                <div class="incident-summary">{escape(str(incident.get('summary') or ''))}</div>
                <div class="incident-meta">{incident_meta_line}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        action_col, note_col = st.columns([1, 2.2], gap="medium")
        with action_col:
            investigate_clicked = st.button(
                "🚀 Launch Autonomous Investigation",
                width="stretch",
                type="primary",
                disabled=inc_status == "RESOLVED",
            )
        with note_col:
            st.caption(
                "The LangGraph agent sweeps lineage, upstream profiles, job telemetry, git diffs "
                "and ad-hoc diagnostics with read-only tools, then drafts a safe fix. Nothing is "
                "written until a human approves."
            )

        if investigate_clicked:
            with st.spinner("Agent gathering forensic evidence…"):
                report, investigate_error = api_post(f"/incidents/{incident['incident_id']}/investigate")
            if report:
                st.session_state["flash"] = (
                    f"Investigation complete · {len(report.get('ranked_causes', []))} ranked causes · "
                    f"{len(report.get('tool_trace', []))} forensic tools"
                )
                st.rerun()
            else:
                st.error(f"Investigation failed: {investigate_error}")

        if investigation:
            st.markdown('<div class="section-title">Agent Synthesis</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="panel" style="color:#cbd5e1;font-size:0.9rem;line-height:1.65;">{escape(str(investigation.get("summary", "")))}</div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="section-title">Ranked Root Causes</div>', unsafe_allow_html=True)
            for cause in investigation.get("ranked_causes", []):
                st.markdown(cause_card(cause), unsafe_allow_html=True)

            impact = investigation.get("downstream_impact") or []
            if impact:
                st.markdown('<div class="section-title">Downstream Impact</div>', unsafe_allow_html=True)
                st.markdown(
                    "".join(f'<span class="cite-chip">{escape(str(asset))}</span>' for asset in impact),
                    unsafe_allow_html=True,
                )

            with st.expander(f"🔬 Agent Forensic Tool Trace · {len(investigation.get('tool_trace', []))} calls", expanded=False):
                for entry in investigation.get("tool_trace", []):
                    state_icon = "✅" if entry.get("ok") else "⚠️"
                    with st.expander(
                        f"{state_icon} {entry.get('tool')} · {entry.get('duration_ms', '—')} ms · args={json.dumps(entry.get('arguments', {}))[:110]}",
                        expanded=False,
                    ):
                        st.json(entry.get("output"))

            st.markdown('<div class="section-title">SQL Remediation Diff</div>', unsafe_allow_html=True)
            sql_left, sql_right = st.columns(2, gap="medium")
            with sql_left:
                if "registration_date IS NOT NULL" in transform_sql:
                    st.caption("⛔ Current on-disk SQL (contains the offending filter)")
                else:
                    st.caption("Current on-disk SQL")
                st.code(transform_sql or "-- transform SQL unavailable", language="sql")
            with sql_right:
                st.caption("✅ Proposed safe SQL")
                st.code(str(investigation.get("proposed_sql") or ""), language="sql")

            st.markdown('<div class="section-title">Rollback Plan</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="rollback-box">{escape(str(investigation.get("rollback_plan") or ""))}</div>',
                unsafe_allow_html=True,
            )

            st.write("")
            approve_col, approve_note = st.columns([1, 2.2], gap="medium")
            with approve_col:
                approve_clicked = st.button(
                    "🔐 Approve & Apply Fix",
                    width="stretch",
                    type="primary",
                    disabled=inc_status == "RESOLVED",
                )
            with approve_note:
                st.caption(
                    "Writes the proposed SQL to sql/transform_daily_customers.sql, re-runs the Day 15 "
                    "ETL against raw_users, re-validates the quality guardrails and closes the incident."
                )

            if approve_clicked:
                with st.spinner("Applying fix and re-running Day 15 ETL…"):
                    approval, approval_error = api_post(
                        f"/incidents/{incident['incident_id']}/approve",
                        {"approved_by": "analytics.lead@sentinel.ai"},
                    )
                if approval:
                    st.session_state["approval"] = approval
                    st.session_state["flash"] = (
                        approval.get("message", "Fix applied"),
                        bool(approval.get("recovered")),
                    )
                    st.rerun()
                else:
                    st.error(f"Approval failed: {approval_error}")

        approval = st.session_state.get("approval")
        if approval and int(approval.get("incident", {}).get("incident_id", -1)) == int(incident["incident_id"]):
            verification = approval.get("verification", {})
            st.markdown('<div class="section-title">Before / After Verification Scorecard</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="scorecard">
                    <div class="score-item" style="background:rgba(239,68,68,0.08);border-color:rgba(239,68,68,0.35);">
                        <div class="label" style="color:#fca5a5;">Rows before fix</div>
                        <div class="value">{fmt(approval.get('rows_before'))}</div>
                        <div class="note">contaminated dim_customers</div>
                    </div>
                    <div class="score-item">
                        <div class="label">Rows after fix</div>
                        <div class="value">{fmt(approval.get('rows_after'))}</div>
                        <div class="note">expected {fmt(approval.get('expected_rows'))}</div>
                    </div>
                    <div class="score-item">
                        <div class="label">Quality guardrails</div>
                        <div class="value">{escape(str(approval.get('quality_status', '—')))}</div>
                        <div class="note">re-run after ETL</div>
                    </div>
                    <div class="score-item">
                        <div class="label">Recovery</div>
                        <div class="value">{'CONFIRMED' if approval.get('recovered') else 'FAILED'}</div>
                        <div class="note">approved by {escape(str(approval.get('approved_by', '—')))}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if verification:
                with st.expander("Verification verdict payload", expanded=False):
                    st.json(verification)

    st.markdown('<div class="section-title">Incident Ledger</div>', unsafe_allow_html=True)
    if incidents:
        st.dataframe(
            pd.DataFrame(incidents)[
                ["incident_id", "table_name", "run_day", "severity", "anomaly_type", "status", "approved_by", "resolved_at"]
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption("No incidents recorded.")

# ------------------------------------------------------------------ Tab 3
with tab_chat:
    st.markdown('<div class="section-title">SentinelAI Data Copilot</div>', unsafe_allow_html=True)
    st.caption(
        "Every answer is grounded in live read-only queries against sentinel.db — lineage, "
        "profiles, job telemetry, git history and ad-hoc diagnostics."
    )

    messages = data.get("chat_messages", []) or []
    if not messages:
        st.markdown(
            """
            <div class="panel">
                <div class="incident-summary">
                    Ask me why <b>dim_customers</b> dropped on Day 15, which dashboards are impacted,
                    for the exact git diff, or how the guest-checkout NULL rate compares to baseline.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for message in messages:
        role = str(message.get("role", "assistant"))
        with st.chat_message("assistant" if role == "assistant" else "user", avatar="🛡️" if role == "assistant" else "🧑‍💻"):
            st.markdown(str(message.get("content", "")))
            if role == "assistant" and message.get("tool_citations_json"):
                try:
                    citations = json.loads(message["tool_citations_json"])
                except json.JSONDecodeError:
                    citations = []
                if citations:
                    with st.expander(f"🔬 Forensic tools consulted · {len(citations)}", expanded=False):
                        st.markdown(tool_pills(citations), unsafe_allow_html=True)
                        for citation in citations:
                            st.caption(f"{citation.get('tool')} {json.dumps(citation.get('arguments', {}))}")

    st.write("")
    st.markdown('<div class="section-title">Suggested Questions</div>', unsafe_allow_html=True)
    chip_columns = st.columns(3, gap="small")
    for idx, prompt_text in enumerate(QUICK_PROMPTS):
        with chip_columns[idx % 3]:
            if st.button(prompt_text, key=f"chip_{idx}", width="stretch"):
                st.session_state["pending_chat"] = prompt_text
                st.rerun()

    pending = st.session_state.pop("pending_chat", None)
    typed = st.chat_input("Ask SentinelAI about pipeline health, root causes or downstream impact…")
    question = typed or pending
    if question:
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="🛡️"):
            with st.spinner("Consulting forensic tools…"):
                response, chat_error = api_post("/chat", {"message": question})
            if response:
                st.markdown(str(response.get("reply", "")))
                citations = response.get("tool_citations", [])
                if citations:
                    with st.expander(f"🔬 Forensic tools consulted · {len(citations)}", expanded=True):
                        st.markdown(tool_pills(citations), unsafe_allow_html=True)
            else:
                st.error(f"Copilot unavailable: {chat_error}")
