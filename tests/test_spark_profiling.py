"""Tests for dscompanion.pipeline.spark_profiling -- the optional Spark-native
full-dataset profiling stage. PySpark and spark_data_profiler are mocked via
sys.modules patching throughout, matching the convention in tests/test_loaders.py --
no real Spark session runs in these tests.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


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


class TestProfileAndSample:
    def _mock_profiler_module(self, mock_profiler_instance):
        mock_module = MagicMock()
        mock_module.SparkDataProfiler.return_value = mock_profiler_instance
        return mock_module

    def test_profiles_full_dataset_and_saves_report(self, tmp_path):
        from dscompanion.pipeline.spark_profiling import profile_and_sample

        mock_spark_df = MagicMock()
        mock_sampled_df = MagicMock()
        mock_sampled_df.toPandas.return_value = pd.DataFrame({"a": [1, 2, 3]})

        mock_profiler_instance = MagicMock()
        mock_profiler_module = self._mock_profiler_module(mock_profiler_instance)

        mock_session = MagicMock()
        mock_session.read.parquet.return_value = mock_spark_df

        modules = _mock_modules(mock_session, has_profiler=True)
        modules["spark_data_profiler"] = mock_profiler_module

        with (
            patch.dict(sys.modules, modules),
            patch(
                "dscompanion.pipeline.spark_profiling._sample_spark_dataframe",
                return_value=mock_sampled_df,
            ) as mock_sample,
        ):
            result = profile_and_sample("some/path", "parquet", tmp_path / "eda")

        mock_profiler_module.SparkDataProfiler.assert_called_once()
        call_kwargs = mock_profiler_module.SparkDataProfiler.call_args.kwargs
        assert call_kwargs["missing_threshold"] == pytest.approx(30.0)
        mock_profiler_instance.profile.assert_called_once()
        mock_profiler_instance.to_html.assert_called_once_with(
            str(tmp_path / "eda" / "spark_profile.html")
        )
        mock_profiler_instance.to_json.assert_called_once_with(
            str(tmp_path / "eda" / "spark_profile.json")
        )
        mock_sample.assert_called_once()
        assert mock_sample.call_args.args[0] is mock_spark_df
        assert isinstance(result, pd.DataFrame)
        assert list(result["a"]) == [1, 2, 3]

    def test_creates_output_dir_if_missing(self, tmp_path):
        from dscompanion.pipeline.spark_profiling import profile_and_sample

        mock_sampled_df = MagicMock()
        mock_sampled_df.toPandas.return_value = pd.DataFrame({"a": [1]})
        mock_session = MagicMock()
        mock_session.read.parquet.return_value = MagicMock()
        modules = _mock_modules(mock_session, has_profiler=True)
        modules["spark_data_profiler"] = self._mock_profiler_module(MagicMock())

        output_dir = tmp_path / "does" / "not" / "exist" / "yet"
        with (
            patch.dict(sys.modules, modules),
            patch(
                "dscompanion.pipeline.spark_profiling._sample_spark_dataframe",
                return_value=mock_sampled_df,
            ),
        ):
            profile_and_sample("some/path", "parquet", output_dir)

        assert output_dir.is_dir()
