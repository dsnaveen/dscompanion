"""Tests for dscompanion.pipeline.spark_profiling -- the optional Spark-native
full-dataset profiling stage. PySpark and spark_data_profiler are mocked via
sys.modules patching throughout, matching the convention in tests/test_loaders.py --
no real Spark session runs in these tests.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch


def _mock_modules(session, has_profiler: bool = True):
    """Build the sys.modules patch dict for a given (session, spark_data_profiler
    presence) combination. session=None simulates "PySpark installed but no active
    session"; omit "pyspark"/"pyspark.sql" entirely from the dict to simulate
    "PySpark not installed" (see test_pyspark_not_installed_returns_false below,
    which patches sys.modules to None instead of using this helper).
    """
    mock_session_cls = MagicMock()
    mock_session_cls.getActiveSession.return_value = session
    mock_pyspark = MagicMock()
    mock_pyspark.sql.SparkSession = mock_session_cls
    modules = {"pyspark": mock_pyspark, "pyspark.sql": mock_pyspark.sql}
    if has_profiler:
        modules["spark_data_profiler"] = MagicMock()
    return modules


class TestSparkProfilingAvailable:
    def test_all_present_returns_true(self):
        from dscompanion.pipeline.spark_profiling import spark_profiling_available

        mock_session = MagicMock()
        with patch.dict(sys.modules, _mock_modules(mock_session, has_profiler=True)):
            assert spark_profiling_available() is True

    def test_pyspark_not_installed_returns_false(self):
        from dscompanion.pipeline.spark_profiling import spark_profiling_available

        with patch.dict(sys.modules, {"pyspark": None, "pyspark.sql": None}):
            assert spark_profiling_available() is False

    def test_no_active_session_returns_false(self):
        from dscompanion.pipeline.spark_profiling import spark_profiling_available

        with patch.dict(sys.modules, _mock_modules(None, has_profiler=True)):
            assert spark_profiling_available() is False

    def test_spark_data_profiler_not_installed_returns_false(self):
        from dscompanion.pipeline.spark_profiling import spark_profiling_available

        mock_session = MagicMock()
        modules = _mock_modules(mock_session, has_profiler=False)
        modules["spark_data_profiler"] = None
        with patch.dict(sys.modules, modules):
            assert spark_profiling_available() is False
