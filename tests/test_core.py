"""Data layer + quality engine tests (stdlib only, no LLM libraries needed)."""

from __future__ import annotations

import pytest

from core.database import query_one, read_transform_sql
from core.quality_engine import recovery_is_confirmed
from data.seed_generator import (
    inject_day15_anomaly,
    reset_and_seed_baseline,
    simulate_day15_healthy,
)


@pytest.fixture(scope="module", autouse=True)
def seeded() -> None:
    reset_and_seed_baseline()


def test_baseline_seed_volumes() -> None:
    row = query_one("SELECT COUNT(*) AS c FROM raw_users")
    assert row["c"] == 1_400_000


def test_healthy_day_passes_quality() -> None:
    result = simulate_day15_healthy()
    assert result["quality_status"] == "PASS"
    assert result["dim_customers_rows"] == 101_200
    assert "IS NOT NULL" not in read_transform_sql()


def test_anomaly_is_detected_as_critical() -> None:
    result = inject_day15_anomaly()
    assert result["dim_customers_rows"] == 62_400
    assert result["quality_status"] == "CRITICAL"
    assert result["incident_id"]


def test_stg_view_exists() -> None:
    assert query_one("SELECT COUNT(*) AS c FROM stg_users_cleaned WHERE run_day = 1")["c"] > 0


def test_recovery_uses_upstream_rows_not_baseline_mean() -> None:
    # A healthy Day 15 is +1.2% vs the 14-day mean; that must still count as recovered.
    simulate_day15_healthy()
    check = recovery_is_confirmed(run_day=15)
    assert check["expected_row_count"] == 101_200
    assert check["recovered"] is True
