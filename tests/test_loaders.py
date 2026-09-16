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


def _mock_spark_df(total_count: int, sampled_frame: pd.DataFrame | None = None):
    mock_df = MagicMock()
    mock_df.count.return_value = total_count
    mock_df.sample.return_value = mock_df
    mock_df.limit.return_value = mock_df
    mock_df.toPandas.return_value = (
        sampled_frame if sampled_frame is not None else pd.DataFrame({"a": range(total_count)})
    )
    return mock_df


class TestLoadDeltaSampling:
    """_load_delta's Spark-side sampling — mocked SparkSession, no real cluster needed."""

    def _mock_df(self, total_count: int, sampled_frame: pd.DataFrame | None = None):
        return _mock_spark_df(total_count, sampled_frame)

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


class TestLoadParquetViaSpark:
    """DataConfig.read_via_spark=True's implementation — same sampling strategy as
    _load_delta, reading via spark.read.parquet() instead of spark.read.format("delta").
    """

    def test_reads_via_spark_parquet_reader(self):
        mock_df = _mock_spark_df(total_count=1000)
        mock_session = MagicMock()
        mock_session.read.parquet.return_value = mock_df

        with patch.dict(sys.modules, _mock_pyspark_modules(mock_session)):
            result = load_raw_data("dbfs:/some/parquet_dir", "parquet", read_via_spark=True)

        mock_session.read.parquet.assert_called_once_with("dbfs:/some/parquet_dir")
        mock_session.read.format.assert_not_called()
        assert len(result) == 1000

    def test_fraction_rows_samples_via_spark_bernoulli(self):
        mock_df = _mock_spark_df(total_count=1000)
        mock_session = MagicMock()
        mock_session.read.parquet.return_value = mock_df

        with patch.dict(sys.modules, _mock_pyspark_modules(mock_session)):
            load_raw_data(
                "dbfs:/some/parquet_dir", "parquet", read_via_spark=True, fraction_rows=0.1
            )

        mock_df.sample.assert_called_once_with(fraction=0.1, seed=settings.random_state)

    def test_nrows_oversamples_and_limits(self):
        mock_df = _mock_spark_df(total_count=1000)
        mock_session = MagicMock()
        mock_session.read.parquet.return_value = mock_df

        with patch.dict(sys.modules, _mock_pyspark_modules(mock_session)):
            load_raw_data("dbfs:/some/parquet_dir", "parquet", read_via_spark=True, nrows=100)

        mock_df.count.assert_called_once()
        mock_df.sample.assert_called_once_with(
            fraction=pytest.approx(0.11), seed=settings.random_state
        )
        mock_df.limit.assert_called_once_with(100)

    def test_no_active_session_raises_runtime_error(self):
        with patch.dict(sys.modules, _mock_pyspark_modules(None)):
            with pytest.raises(RuntimeError, match="No active SparkSession"):
                load_raw_data("dbfs:/some/parquet_dir", "parquet", read_via_spark=True)

    def test_import_error_raises_runtime_error(self):
        with patch.dict(sys.modules, {"pyspark": None, "pyspark.sql": None}):
            with pytest.raises(RuntimeError, match="requires PySpark"):
                load_raw_data("dbfs:/some/parquet_dir", "parquet", read_via_spark=True)


class TestLoadParquetViaRowGroups:
    """DataConfig.row_group_sample=True's implementation — real pyarrow, no mocking needed."""

    @pytest.fixture
    def multi_row_group_parquet(self, tmp_path):
        """Write several small parquet files (one per "row group" for sampling purposes) into
        a directory, mimicking a partitioned dataset with many fragments to sample across.
        """
        directory = tmp_path / "dataset"
        directory.mkdir()
        for i in range(20):
            chunk = pd.DataFrame({"idx": range(i * 50, (i + 1) * 50)})
            chunk.to_parquet(directory / f"part_{i}.parquet", index=False)
        return directory

    def test_no_sampling_returns_full_dataset(self, multi_row_group_parquet):
        result = load_raw_data(str(multi_row_group_parquet), "parquet", row_group_sample=True)
        assert len(result) == 1000

    def test_nrows_returns_approximately_requested_count(self, multi_row_group_parquet):
        result = load_raw_data(
            str(multi_row_group_parquet), "parquet", row_group_sample=True, nrows=100
        )
        assert len(result) == 100

    def test_nrows_sample_is_random_not_sequential(self, multi_row_group_parquet):
        result = load_raw_data(
            str(multi_row_group_parquet), "parquet", row_group_sample=True, nrows=100
        )
        assert sorted(result["idx"].tolist()) != list(range(100))

    def test_reproducible_via_settings_random_state(self, multi_row_group_parquet):
        first = load_raw_data(
            str(multi_row_group_parquet), "parquet", row_group_sample=True, nrows=100
        )
        second = load_raw_data(
            str(multi_row_group_parquet), "parquet", row_group_sample=True, nrows=100
        )
        assert sorted(first["idx"].tolist()) == sorted(second["idx"].tolist())

    def test_fraction_rows_returns_approximately_requested_fraction(self, multi_row_group_parquet):
        result = load_raw_data(
            str(multi_row_group_parquet), "parquet", row_group_sample=True, fraction_rows=0.1
        )
        assert 90 <= len(result) <= 110

    def test_nrows_exceeding_dataset_size_returns_full_dataset(self, multi_row_group_parquet):
        result = load_raw_data(
            str(multi_row_group_parquet), "parquet", row_group_sample=True, nrows=10_000
        )
        assert len(result) == 1000

    def test_single_file_still_works(self, tmp_path):
        df = pd.DataFrame({"idx": range(500)})
        path = tmp_path / "single.parquet"
        df.to_parquet(path, index=False)
        result = load_raw_data(str(path), "parquet", row_group_sample=True, nrows=50)
        assert len(result) == 50
