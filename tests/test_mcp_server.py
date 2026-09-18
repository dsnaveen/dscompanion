"""Tests for dscompanion.mcp -- the local MCP server exposing dscompanion as
agent-callable tools. All tests call the underlying functions directly (not
through an actual MCP client/transport), per this project's established
small-fixture testing convention (see tests/test_chart_export.py).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dscompanion.config import settings


@pytest.fixture(autouse=True)
def _isolated_mcp_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "mcp_output_dir", str(tmp_path / "mcp_outputs"))


class TestArtifacts:
    def test_new_run_dir_creates_unique_directory(self):
        from dscompanion.mcp._artifacts import new_run_dir

        run_dir = new_run_dir()
        assert run_dir.exists()
        assert run_dir.is_dir()
        assert str(run_dir).startswith(settings.mcp_output_dir)

    def test_new_run_dir_returns_different_paths_each_call(self):
        from dscompanion.mcp._artifacts import new_run_dir

        assert new_run_dir() != new_run_dir()

    def test_save_and_load_split_round_trip(self):
        from dscompanion.mcp._artifacts import load_split, new_run_dir, save_split
        from dscompanion.split import DataSplitter

        df = pd.DataFrame({"x": range(20), "target": [0, 1] * 10})
        split = DataSplitter(strategy="random", target_col="target").fit_split(df)
        path = new_run_dir() / "split.joblib"

        saved_path = save_split(split, path)
        loaded = load_split(saved_path)

        assert Path(saved_path).exists()
        assert loaded.train_X.equals(split.train_X)
        assert loaded.train_y.equals(split.train_y)


class TestLoading:
    def test_load_dataframe_csv(self, tmp_path):
        from dscompanion.mcp._loading import load_dataframe

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        path = tmp_path / "data.csv"
        df.to_csv(path, index=False)

        loaded = load_dataframe(str(path))

        assert list(loaded.columns) == ["a", "b"]
        assert len(loaded) == 3

    def test_load_dataframe_parquet(self, tmp_path):
        from dscompanion.mcp._loading import load_dataframe

        df = pd.DataFrame({"a": [1, 2, 3]})
        path = tmp_path / "data.parquet"
        df.to_parquet(path, index=False)

        loaded = load_dataframe(str(path))

        assert list(loaded.columns) == ["a"]

    def test_load_dataframe_unsupported_extension_raises(self, tmp_path):
        from dscompanion.mcp._loading import load_dataframe

        path = tmp_path / "data.txt"
        path.write_text("not a dataset")

        with pytest.raises(ValueError, match="[Uu]nsupported"):
            load_dataframe(str(path))
