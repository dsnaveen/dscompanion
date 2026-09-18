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


class TestAnalyzeDataset:
    def _write_csv(self, tmp_path) -> str:
        df = pd.DataFrame(
            {
                "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70] * 3,
                "balance": [float(i * 100) for i in range(1, 11)] * 3,
                "target": [0, 1] * 15,
            }
        )
        path = tmp_path / "data.csv"
        df.to_csv(path, index=False)
        return str(path)

    def test_returns_summary_and_charts(self, tmp_path):
        from dscompanion.mcp.server import analyze_dataset

        data_path = self._write_csv(tmp_path)

        result = analyze_dataset(data_path, target="target")

        assert "error" not in result
        assert result["summary"]["n_rows"] == 30
        assert "age" in result["summary"]["numeric_columns"]
        assert Path(result["run_dir"]).exists()
        assert len(result["chart_paths"]) > 0

    def test_missing_target_column_returns_error(self, tmp_path):
        from dscompanion.mcp.server import analyze_dataset

        data_path = self._write_csv(tmp_path)

        result = analyze_dataset(data_path, target="does_not_exist")

        assert "error" in result

    def test_missing_file_returns_error(self, tmp_path):
        from dscompanion.mcp.server import analyze_dataset

        result = analyze_dataset(str(tmp_path / "nope.csv"), target="target")

        assert "error" in result


class TestTrainAndCompareModels:
    def _write_csv(self, tmp_path) -> str:
        import numpy as np

        rng = np.random.RandomState(42)
        n = 200
        df = pd.DataFrame(
            {
                "x1": rng.normal(size=n),
                "x2": rng.normal(size=n),
                "target": rng.choice([0, 1], size=n),
            }
        )
        path = tmp_path / "data.csv"
        df.to_csv(path, index=False)
        return str(path)

    def test_trains_and_returns_leaderboard(self, tmp_path):
        from dscompanion.mcp.server import train_and_compare_models

        data_path = self._write_csv(tmp_path)

        result = train_and_compare_models(data_path, target="target", task="classification")

        assert "error" not in result
        assert Path(result["run_dir"]).exists()
        assert Path(result["model_path"]).exists()
        assert Path(result["run_dir"], "split.joblib").exists()
        assert len(result["leaderboard"]) > 0
        assert result["recommended"] in {row["algorithm"] for row in result["leaderboard"]}

    def test_non_classification_task_returns_error(self, tmp_path):
        from dscompanion.mcp.server import train_and_compare_models

        data_path = self._write_csv(tmp_path)

        result = train_and_compare_models(data_path, target="target", task="regression")

        assert "error" in result
        assert "classification" in result["error"]

    def test_missing_target_column_returns_error(self, tmp_path):
        from dscompanion.mcp.server import train_and_compare_models

        data_path = self._write_csv(tmp_path)

        result = train_and_compare_models(data_path, target="nope", task="classification")

        assert "error" in result


class TestReadinessChecks:
    def _fitted_model_and_split(self):
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        from dscompanion.models import ClassificationModel
        from dscompanion.split import DataSplitter

        rng = np.random.RandomState(0)
        n = 200
        df = pd.DataFrame(
            {
                "x1": rng.normal(size=n),
                "x2": rng.normal(size=n),
                "target": rng.choice([0, 1], size=n),
            }
        )
        split = DataSplitter(strategy="random", target_col="target").fit_split(df)
        model = ClassificationModel(estimator=LogisticRegression())
        model.fit(split.train_X, split.train_y)
        return model, split

    def test_returns_four_checks(self):
        from dscompanion.mcp._readiness import readiness_checks

        model, split = self._fitted_model_and_split()

        checks = readiness_checks(model, split)

        names = {c["name"] for c in checks}
        assert names == {"leakage", "near_random", "overfitting_gap", "psi"}
        assert all(c["status"] in {"pass", "warn", "fail"} for c in checks)


class TestCheckModelReadiness:
    def _trained_run_dir(self, tmp_path):
        import numpy as np

        from dscompanion.mcp.server import train_and_compare_models

        rng = np.random.RandomState(1)
        n = 200
        df = pd.DataFrame(
            {
                "x1": rng.normal(size=n),
                "x2": rng.normal(size=n),
                "target": rng.choice([0, 1], size=n),
            }
        )
        data_path = tmp_path / "data.csv"
        df.to_csv(data_path, index=False)
        result = train_and_compare_models(str(data_path), target="target", task="classification")
        return result["run_dir"]

    def test_returns_checklist_and_overall(self, tmp_path):
        from dscompanion.mcp.server import check_model_readiness

        run_dir = self._trained_run_dir(tmp_path)

        result = check_model_readiness(run_dir)

        assert "error" not in result
        assert len(result["checks"]) == 4
        assert result["overall"] in {"ready", "needs_review", "not_ready"}

    def test_missing_run_dir_returns_error(self, tmp_path):
        from dscompanion.mcp.server import check_model_readiness

        result = check_model_readiness(str(tmp_path / "nonexistent"))

        assert "error" in result


class TestExplainModel:
    def _trained_run_dir(self, tmp_path):
        import numpy as np

        from dscompanion.mcp.server import train_and_compare_models

        rng = np.random.RandomState(2)
        n = 200
        df = pd.DataFrame(
            {
                "x1": rng.normal(size=n),
                "x2": rng.normal(size=n),
                "target": rng.choice([0, 1], size=n),
            }
        )
        data_path = tmp_path / "data.csv"
        df.to_csv(data_path, index=False)
        result = train_and_compare_models(str(data_path), target="target", task="classification")
        return result["run_dir"]

    def test_returns_top_features(self, tmp_path):
        from dscompanion.mcp.server import explain_model

        run_dir = self._trained_run_dir(tmp_path)

        result = explain_model(run_dir)

        assert "error" not in result
        assert len(result["top_features"]) == 2
        assert {"feature", "mean_abs_shap", "rank"} <= set(result["top_features"][0].keys())

    def test_missing_run_dir_returns_error(self, tmp_path):
        from dscompanion.mcp.server import explain_model

        result = explain_model(str(tmp_path / "nonexistent"))

        assert "error" in result


class TestGenerateReport:
    def _trained_run_dir(self, tmp_path):
        import numpy as np

        from dscompanion.mcp.server import train_and_compare_models

        rng = np.random.RandomState(3)
        n = 200
        df = pd.DataFrame(
            {
                "x1": rng.normal(size=n),
                "x2": rng.normal(size=n),
                "target": rng.choice([0, 1], size=n),
            }
        )
        data_path = tmp_path / "data.csv"
        df.to_csv(data_path, index=False)
        result = train_and_compare_models(str(data_path), target="target", task="classification")
        return result["run_dir"]

    def test_returns_excel_and_html_paths(self, tmp_path):
        from dscompanion.mcp.server import generate_report

        run_dir = self._trained_run_dir(tmp_path)

        result = generate_report(run_dir)

        assert "error" not in result
        assert Path(result["excel_path"]).exists()
        assert Path(result["html_path"]).exists()

    def test_missing_run_dir_returns_error(self, tmp_path):
        from dscompanion.mcp.server import generate_report

        result = generate_report(str(tmp_path / "nonexistent"))

        assert "error" in result
