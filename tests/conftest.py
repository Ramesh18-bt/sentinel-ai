"""Test isolation: every run uses a throwaway database and SQL file.

The environment variables must be set before ``core.database`` is imported,
because it resolves its paths at import time.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="sentinel-tests-"))
os.environ["SENTINEL_DB_PATH"] = str(_TMP / "sentinel_test.db")
os.environ["SENTINEL_TRANSFORM_SQL_PATH"] = str(_TMP / "transform_daily_customers.sql")
os.environ["SENTINEL_MOCK_MODE"] = "true"
os.environ["SENTINEL_AUTO_SEED"] = "false"
os.environ.pop("OPENAI_API_KEY", None)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
