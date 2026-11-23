# SentinelAI

**Autonomous data quality, root cause investigation, and an interactive data copilot.**

SentinelAI watches a daily customer-revenue pipeline, detects a silent data failure that
the orchestrator happily reports as `SUCCESS`, autonomously investigates it with read-only
forensic tools, drafts a safe fix, and applies it through a human-in-the-loop approval gate.

Runs completely offline by default — real SQLite, real SQL, real statistical guardrails, no API key.

---

## Architecture

```text
                        ┌──────────────────────────────────────────────┐
                        │        Streamlit Command Center (ui/)        │
                        │  telemetry · lineage · HITL · copilot chat   │
                        └───────────────────────┬──────────────────────┘
                                                │ HTTP (httpx)
                        ┌───────────────────────▼──────────────────────┐
                        │           FastAPI closed loop (api/)         │
                        │ reset · simulate · investigate · approve     │
                        │ dashboard-data · chat                        │
                        └──────┬───────────────────────┬───────────────┘
                               │                       │
              ┌────────────────▼──────────┐   ┌────────▼─────────────────┐
              │ LangGraph agent (agent/)  │   │ seed simulator (data/)   │
              │ gather → analyze → draft  │   │ baseline + Day 15 fault  │
              └────────────┬──────────────┘   └────────┬─────────────────┘
                           │  @tool (read-only)         │
              ┌────────────▼──────────────┐            │
              │ forensic tools (core/)    │            │
              │ lineage · profile · logs  │            │
              │ git diff · SELECT-only SQL│            │
              └────────────┬──────────────┘            │
                           │                           │
                        ┌──▼───────────────────────────▼──┐
                        │      sentinel.db (SQLite, WAL)   │
                        │ raw_users · dim_customers ·      │
                        │ pipeline_runs · metric_history · │
                        │ lineage_edges · code_commits ·   │
                        │ incidents · chat_messages        │
                        └──────────────────────────────────┘
```

Data flow: `raw_users → stg_users_cleaned (ingest_users) → dim_customers
(transform_daily_customers) → exec_revenue_dashboard (build_exec_mart)`.

### The demo storyline

| Stage | What happens |
| --- | --- |
| Days 1–14 | 14 daily partitions, ~100k rows each, 12% guest checkouts (`registration_date IS NULL`), `COALESCE` fallback keeps every customer. |
| Day 15 healthy | 101,200 rows in, 101,200 rows out. Quality engine passes. |
| Day 15 anomaly | Promo pushes guest checkouts to **38.3% NULL**. Commit **`a8f92c1`** (alex.dev) adds `AND registration_date IS NOT NULL` to the transform. `dim_customers` silently lands at **62,400 rows (-37.6%)**, and `pipeline_runs.status` still says `SUCCESS`. |
| Detection | `quality_engine.py` compares against the 14-day rolling mean/std-dev and raises a `CRITICAL` incident (row-count drop > 15%, z-score < -3, NULL-rate spike > 10pp). |
| Investigation | The agent calls the 5 read-only tools, ranks causes with confidence scores and citations, proposes the corrected SQL + rollback plan. |
| Approval | One click rewrites `sql/transform_daily_customers.sql`, re-runs the Day 15 ETL, re-validates the guardrails (101,200 rows restored), and resolves the incident. |

---

## Quickstart (uv, Windows Git Bash / PowerShell / Linux)

```bash
cd sentinel-ai

uv sync                                         # create .venv + install deps

uv run python -m data.seed_generator            # seed Days 1–14 + healthy Day 15
uv run uvicorn api.main:app --port 8000         # terminal 1: backend
uv run streamlit run ui/app.py                  # terminal 2: dashboard
```

Open <http://localhost:8501>. The API seeds the baseline automatically on first boot, so the
dashboard is never empty even if you skip the seed step.

### Simulation from the CLI

```bash
uv run python -m data.seed_generator --mode baseline   # reset Days 1–14 only
uv run python -m data.seed_generator --mode healthy    # clean Day 15
uv run python -m data.seed_generator --mode anomaly    # inject the failure
```

### Demo walkthrough

1. **Sidebar → 1. Reset Baseline (Days 1–14)** — telemetry chart populates.
2. **Sidebar → 2. Run Healthy Day 15** — green status pill, quality `PASS`.
3. **Sidebar → 3. Inject Day 15 Anomaly** — crimson pulsing status, incident #1 raised.
4. **Tab 2 → Launch Autonomous Investigation** — ranked causes, tool trace, SQL diff, rollback plan.
5. **Approve & Apply Fix** — before/after scorecard, 62,400 → 101,200 rows, `RECOVERED`.
6. **Tab 3** — ask *“Why did dim_customers drop on Day 15?”* and inspect the tools it consulted.

---

## Configuration

Copy `.env.example` to `.env`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENAI_API_KEY` | *(empty)* | Enables live LLM mode. |
| `SENTINEL_MOCK_MODE` | `true` | `true` = deterministic offline engine. |
| `SENTINEL_DB_PATH` | `sentinel.db` | SQLite file location. |
| `SENTINEL_API_URL` | `http://localhost:8000` | Backend URL used by the UI. |

Live mode requires **both** `SENTINEL_MOCK_MODE=false` **and** a non-empty `OPENAI_API_KEY`.
Otherwise SentinelAI runs the offline deterministic engine, which still executes every real
`@tool`, every real SQL query and every statistical guardrail — it only skips the LLM call.

---

## API surface

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/reset` | Reseed Days 1–14, restore healthy SQL, clear chat. |
| `POST` | `/simulate/healthy` | Land clean Day 15. |
| `POST` | `/simulate/anomaly` | Inject the silent Day 15 failure. |
| `GET` | `/dashboard-data` | Metrics, runs, lineage, commits, incidents, chat + KPIs. |
| `POST` | `/incidents/{id}/investigate` | Run the LangGraph agent, persist the report. |
| `POST` | `/incidents/{id}/approve` | Apply `proposed_sql`, re-run ETL, verify recovery, resolve. |
| `POST` | `/chat` | Data Copilot turn with tool citations. |

Interactive docs: <http://localhost:8000/docs>.

---

## Project layout

```text
sentinel-ai/
├── pyproject.toml                          # uv project, Python >= 3.11
├── .env.example
├── README.md
├── sql/transform_daily_customers.sql       # the real, rewritable transform
├── data/seed_generator.py                  # deterministic baseline + fault injection
├── core/database.py                        # schema contract, WAL, SQL execution
├── core/quality_engine.py                  # 14-day rolling guardrails
├── core/tools.py                           # 5 read-only @tool forensics
├── agent/state.py                          # Pydantic report + graph state
├── agent/graph.py                          # gather → analyze → draft
├── agent/chat_copilot.py                   # copilot (offline routing / live tools)
├── api/main.py                             # FastAPI closed loop
└── ui/app.py                               # Streamlit dark observability UI
```

## Safety model

* Every forensic tool is read-only; `run_readonly_diagnostic_sql` rejects anything that is not
  a single `SELECT`/`WITH` statement and blocks 17 DDL/DML keywords.
* The agent can only *propose* SQL. Writing files, re-running ETL and resolving incidents
  require an explicit `POST /approve` with an `approved_by` identity.
* Applying a proposal is refused if it still contains `registration_date IS NOT NULL`.
* Rollback is always a `git revert` of the correlated commit plus a re-run and a row-count assertion.
