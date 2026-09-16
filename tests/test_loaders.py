"""Tests for dscompanion.pipeline.loaders — shared raw-file loading."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from dscompanion.config import settings
from dscompanion.pipeline.loaders import load_raw_data


class TestLoadRawDataDispatch:
    def test_parquet(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3]})
        path = tmp_path / "x.parquet"
        df.to_parquet(path, index=False)
        loaded = load_raw_data(str(path), "parquet")
        pd.testing.assert_frame_equal(loaded, df)

    def test_csv(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3]})
        path = tmp_path / "x.csv"
        df.to_csv(path, index=False)
        loaded = load_raw_data(str(path), "csv")
        pd.testing.assert_frame_equal(loaded, df)

    def test_unsupported_format_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="Unsupported format"):
            load_raw_data(str(tmp_path / "x.xml"), "xml")

    def test_missing_file_raises_runtime_error(self, tmp_path):
        with pytest.raises(RuntimeError, match="Failed to load data"):
            load_raw_data(str(tmp_path / "does_not_exist.parquet"), "parquet")

    def test_nrows_and_fraction_rows_applied_for_parquet(self, tmp_path):
        df = pd.DataFrame({"idx": range(1000)})
        path = tmp_path / "x.parquet"
        df.to_parquet(path, index=False)
        assert len(load_raw_data(str(path), "parquet", nrows=10)) == 10
        assert len(load_raw_data(str(path), "parquet", fraction_rows=0.1)) == 100


def _mock_pyspark_modules(mock_session):
    """Build the {"pyspark": ..., "pyspark.sql": ...} dict for patch.dict(sys.modules, ...)."""
    mock_session_cls = MagicMock()
    mock_session_cls.getActiveSession.return_value = mock_session
    mock_pyspark = MagicMock()
    mock_pyspark.sql.SparkSession = mock_session_cls
    return {"pyspark": mock_pyspark, "pyspark.sql": mock_pyspark.sql}


class TestLoadDeltaSampling:
    """_load_delta's Spark-side sampling — mocked SparkSession, no real cluster needed."""

    def _mock_df(self, total_count: int, sampled_frame: pd.DataFrame | None = None):
        mock_df = MagicMock()
        mock_df.count.return_value = total_count
        mock_df.sample.return_value = mock_df
        mock_df.limit.return_value = mock_df
        mock_df.toPandas.return_value = (
            sampled_frame if sampled_frame is not None else pd.DataFrame({"a": range(total_count)})
        )
        return mock_df

    def test_no_sampling_when_nrows_and_fraction_rows_unset(self):
        mock_df = self._mock_df(total_count=1000)
        mock_session = MagicMock()
        mock_session.read.format.return_value.load.return_value = mock_df

        with patch.dict(sys.modules, _mock_pyspark_modules(mock_session)):
            result = load_raw_data("dbfs:/some/table", "delta")

        mock_df.sample.assert_not_called()
        mock_df.count.assert_not_called()
        mock_df.limit.assert_not_called()
        mock_df.toPandas.assert_called_once()
        assert len(result) == 1000

    def test_fraction_rows_samples_via_spark_bernoulli(self):
        mock_df = self._mock_df(total_count=1000)
        mock_session = MagicMock()
        mock_session.read.format.return_value.load.return_value = mock_df

        with patch.dict(sys.modules, _mock_pyspark_modules(mock_session)):
            load_raw_data("dbfs:/some/table", "delta", fraction_rows=0.1)

        mock_df.sample.assert_called_once_with(fraction=0.1, seed=settings.random_state)
        mock_df.count.assert_not_called()
        mock_df.limit.assert_not_called()

    def test_nrows_less_than_total_oversamples_and_limits(self):
        mock_df = self._mock_df(total_count=1000)
        mock_session = MagicMock()
        mock_session.read.format.return_value.load.return_value = mock_df

        with patch.dict(sys.modules, _mock_pyspark_modules(mock_session)):
            load_raw_data("dbfs:/some/table", "delta", nrows=100)

        mock_df.count.assert_called_once()
        mock_df.sample.assert_called_once_with(
            fraction=pytest.approx(0.11), seed=settings.random_state
        )
        mock_df.limit.assert_called_once_with(100)

    def test_nrows_oversample_fraction_capped_at_one(self):
        mock_df = self._mock_df(total_count=100)
        mock_session = MagicMock()
        mock_session.read.format.return_value.load.return_value = mock_df

        with patch.dict(sys.modules, _mock_pyspark_modules(mock_session)):
            load_raw_data("dbfs:/some/table", "delta", nrows=95)

        # nrows/total * 1.1 would exceed 1.0 — must be capped, not passed straight to Spark.
        called_fraction = mock_df.sample.call_args.kwargs["fraction"]
        assert called_fraction <= 1.0

    def test_nrows_greater_than_or_equal_to_total_skips_sampling(self):
        mock_df = self._mock_df(total_count=50)
        mock_session = MagicMock()
        mock_session.read.format.return_value.load.return_value = mock_df

        with patch.dict(sys.modules, _mock_pyspark_modules(mock_session)):
            load_raw_data("dbfs:/some/table", "delta", nrows=10_000)

        mock_df.count.assert_called_once()
        mock_df.sample.assert_not_called()
        mock_df.limit.assert_not_called()

    def test_no_active_session_raises_runtime_error(self):
        with patch.dict(sys.modules, _mock_pyspark_modules(None)):
            with pytest.raises(RuntimeError, match="No active SparkSession"):
                load_raw_data("dbfs:/some/table", "delta")

    def test_import_error_raises_runtime_error(self):
        with patch.dict(sys.modules, {"pyspark": None, "pyspark.sql": None}):
            with pytest.raises(RuntimeError, match="requires PySpark"):
                load_raw_data("dbfs:/some/table", "delta")
