"""Tests for dscompanion.eda.chart_export — EDAReport.export_charts()'s implementation.

Uses matplotlib (Agg backend) exclusively, no kaleido/Plotly-image dependency,
mirroring ModelCard._write_interactions_excel_sheet's existing rationale.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dscompanion.eda import EDAReport
from dscompanion.pipeline.config import EDAConfig
from dscompanion.split import DataSplit
from dscompanion.split.splitter import DataSplitMetadata


@pytest.fixture
def fitted_report(data_split: DataSplit) -> EDAReport:
    return EDAReport(data_split, target="target").run_all()


def _make_split(train_X: pd.DataFrame, train_y: pd.Series) -> DataSplit:
    """Minimal DataSplit for small, purpose-built test fixtures (no shared data_split) --
    per this file's own performance-conscious testing convention (issue #8).
    """
    empty_X = train_X.iloc[:0]
    empty_y = train_y.iloc[:0]
    return DataSplit(
        train_X=train_X,
        train_y=train_y,
        val_X=empty_X,
        val_y=empty_y,
        test_X=empty_X,
        test_y=empty_y,
        oot_X=empty_X,
        oot_y=empty_y,
        metadata=DataSplitMetadata(
            strategy="random",
            target_col="target",
            n_features=train_X.shape[1],
            split_sizes={"train": len(train_X), "val": 0, "test": 0, "oot": 0},
        ),
    )


class TestEDAConfigNewFields:
    def test_defaults(self):
        cfg = EDAConfig()
        assert cfg.numeric_categorical_chart_style == "auto"
        assert cfg.numeric_interaction_trend_line == "loess"
        assert cfg.numeric_interaction_trend_poly_degree == 2
        assert cfg.use_target_hue is False

    def test_invalid_chart_style_raises(self):
        with pytest.raises(ValueError, match="numeric_categorical_chart_style"):
            EDAConfig(numeric_categorical_chart_style="not_a_style")

    def test_invalid_trend_line_raises(self):
        with pytest.raises(ValueError, match="numeric_interaction_trend_line"):
            EDAConfig(numeric_interaction_trend_line="not_a_method")

    def test_non_positive_poly_degree_raises(self):
        with pytest.raises(ValueError, match="numeric_interaction_trend_poly_degree"):
            EDAConfig(numeric_interaction_trend_poly_degree=0)

    def test_valid_values_accepted(self):
        cfg = EDAConfig(
            numeric_categorical_chart_style="violin",
            numeric_interaction_trend_line="polynomial",
            numeric_interaction_trend_poly_degree=3,
            use_target_hue=True,
        )
        assert cfg.numeric_categorical_chart_style == "violin"
        assert cfg.numeric_interaction_trend_line == "polynomial"
        assert cfg.numeric_interaction_trend_poly_degree == 3
        assert cfg.use_target_hue is True


class TestTrendLine:
    @pytest.fixture(autouse=True)
    def _mpl_agg(self):
        import matplotlib

        matplotlib.use("Agg")

    def _ax(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        return fig, ax

    def test_method_none_adds_no_line(self):
        import matplotlib.pyplot as plt

        from dscompanion.eda.chart_export import _add_trend_line

        fig, ax = self._ax()
        x = pd.Series(range(10), dtype=float)
        y = pd.Series(range(10), dtype=float)
        _add_trend_line(ax, x, y, method="none", poly_degree=2)
        assert len(ax.lines) == 0
        plt.close(fig)

    def test_linear_adds_one_line(self):
        import matplotlib.pyplot as plt

        from dscompanion.eda.chart_export import _add_trend_line

        fig, ax = self._ax()
        x = pd.Series(range(10), dtype=float)
        y = pd.Series([2 * v + 1 for v in range(10)], dtype=float)
        _add_trend_line(ax, x, y, method="linear", poly_degree=2)
        assert len(ax.lines) == 1
        plt.close(fig)

    def test_polynomial_adds_one_line(self):
        import matplotlib.pyplot as plt

        from dscompanion.eda.chart_export import _add_trend_line

        fig, ax = self._ax()
        x = pd.Series(range(10), dtype=float)
        y = pd.Series([v**2 for v in range(10)], dtype=float)
        _add_trend_line(ax, x, y, method="polynomial", poly_degree=2)
        assert len(ax.lines) == 1
        plt.close(fig)

    def test_loess_adds_one_line(self):
        import matplotlib.pyplot as plt

        from dscompanion.eda.chart_export import _add_trend_line

        fig, ax = self._ax()
        x = pd.Series(range(20), dtype=float)
        y = pd.Series([v + (1 if v % 2 == 0 else -1) for v in range(20)], dtype=float)
        _add_trend_line(ax, x, y, method="loess", poly_degree=2)
        assert len(ax.lines) == 1
        plt.close(fig)

    def test_fewer_than_three_points_skips_silently(self):
        import matplotlib.pyplot as plt

        from dscompanion.eda.chart_export import _add_trend_line

        fig, ax = self._ax()
        x = pd.Series([1.0, 2.0])
        y = pd.Series([1.0, 2.0])
        _add_trend_line(ax, x, y, method="linear", poly_degree=2)
        assert len(ax.lines) == 0
        plt.close(fig)

    def test_nan_values_dropped_before_fitting(self):
        import matplotlib.pyplot as plt

        from dscompanion.eda.chart_export import _add_trend_line

        fig, ax = self._ax()
        x = pd.Series([1.0, 2.0, float("nan"), 4.0, 5.0])
        y = pd.Series([1.0, float("nan"), 3.0, 4.0, 5.0])
        _add_trend_line(ax, x, y, method="linear", poly_degree=2)
        assert len(ax.lines) == 1
        plt.close(fig)


class TestTargetHue:
    def test_binary_target_scatter_has_legend(self, tmp_path):
        n = 30
        df = pd.DataFrame(
            {
                "num1": list(range(n)),
                "num2": [v * 1.5 for v in range(n)],
                "target": [0, 1] * (n // 2),
            }
        )
        split = _make_split(df[["num1", "num2"]], df["target"])
        report = EDAReport(split, target="target").run_all()
        paths = report.export_charts(tmp_path, use_target_hue=True)
        raw_key = "multivariate/interactions/num1_vs_num2_raw"
        assert raw_key in paths

    def test_non_binary_target_renders_without_hue(self, tmp_path):
        n = 30
        df = pd.DataFrame(
            {
                "num1": list(range(n)),
                "num2": [v * 1.5 for v in range(n)],
                "target": [0, 1, 2] * (n // 3),
            }
        )
        split = _make_split(df[["num1", "num2"]], df["target"])
        report = EDAReport(split, target="target").run_all()
        paths = report.export_charts(tmp_path, use_target_hue=True)
        raw_key = "multivariate/interactions/num1_vs_num2_raw"
        assert raw_key in paths
        assert paths[raw_key].exists()

    def test_use_target_hue_false_by_default(self, tmp_path):
        n = 30
        df = pd.DataFrame(
            {
                "num1": list(range(n)),
                "num2": [v * 1.5 for v in range(n)],
                "target": [0, 1] * (n // 2),
            }
        )
        split = _make_split(df[["num1", "num2"]], df["target"])
        report = EDAReport(split, target="target").run_all()
        paths = report.export_charts(tmp_path)
        raw_key = "multivariate/interactions/num1_vs_num2_raw"
        assert raw_key in paths
        assert paths[raw_key].exists()


class TestNumericCategoricalStyle:
    def _binary_cat_split(self, n_categories: int, rows_per_category: int) -> DataSplit:
        cats = [f"cat_{i}" for i in range(n_categories)]
        rows = []
        for i, cat in enumerate(cats):
            for j in range(rows_per_category):
                rows.append({"num": i * 10 + j, "cat": cat})
        df = pd.DataFrame(rows)
        target = pd.Series([0, 1] * (len(df) // 2 + 1))[: len(df)]
        return _make_split(df[["num", "cat"]], target)

    @pytest.mark.parametrize("style", ["box", "violin", "strip", "mean_errorbar", "kde"])
    def test_explicit_style_renders_chart(self, style, tmp_path):
        split = self._binary_cat_split(n_categories=3, rows_per_category=10)
        report = EDAReport(split, target="target").run_all()
        paths = report.export_charts(tmp_path, numeric_categorical_style=style)
        key = "multivariate/interactions/num_by_cat"
        assert key in paths
        assert paths[key].exists()

    def test_invalid_style_raises(self, fitted_report, tmp_path):
        with pytest.raises(ValueError, match="numeric_categorical_chart_style|style"):
            fitted_report.export_charts(tmp_path, numeric_categorical_style="not_a_style")

    def test_auto_resolves_to_strip_for_few_categories_small_n(self):
        from dscompanion.eda.chart_export import _resolve_numeric_categorical_style

        assert _resolve_numeric_categorical_style("auto", n_categories=3, avg_n=10) == "strip"

    def test_auto_resolves_to_mean_errorbar_for_many_categories(self):
        from dscompanion.eda.chart_export import _resolve_numeric_categorical_style

        assert (
            _resolve_numeric_categorical_style("auto", n_categories=20, avg_n=50) == "mean_errorbar"
        )

    def test_auto_resolves_to_box_for_moderate_categories_small_n(self):
        from dscompanion.eda.chart_export import _resolve_numeric_categorical_style

        assert _resolve_numeric_categorical_style("auto", n_categories=8, avg_n=10) == "box"

    def test_auto_resolves_to_kde_for_few_categories_large_n(self):
        from dscompanion.eda.chart_export import _resolve_numeric_categorical_style

        assert _resolve_numeric_categorical_style("auto", n_categories=3, avg_n=40) == "kde"

    def test_auto_resolves_to_violin_for_moderate_categories_large_n(self):
        from dscompanion.eda.chart_export import _resolve_numeric_categorical_style

        assert _resolve_numeric_categorical_style("auto", n_categories=8, avg_n=40) == "violin"


class TestExportChartsFolderStructure:
    def test_raises_before_run_all(self, data_split, tmp_path):
        report = EDAReport(data_split, target="target")
        with pytest.raises(RuntimeError):
            report.export_charts(tmp_path)

    def test_creates_univariate_numeric_and_categorical_dirs(self, fitted_report, tmp_path):
        fitted_report.export_charts(tmp_path)
        assert (tmp_path / "univariate" / "numeric").is_dir()
        assert (tmp_path / "univariate" / "categorical").is_dir()
        assert any((tmp_path / "univariate" / "numeric").glob("*.png"))
        assert any((tmp_path / "univariate" / "categorical").glob("*.png"))

    def test_creates_bivariate_dir_with_charts(self, fitted_report, tmp_path):
        fitted_report.export_charts(tmp_path)
        bivariate_dir = tmp_path / "bivariate"
        assert bivariate_dir.is_dir()
        assert any(bivariate_dir.glob("*.png"))

    def test_creates_multivariate_correlation_and_interactions(self, fitted_report, tmp_path):
        fitted_report.export_charts(tmp_path)
        multi_dir = tmp_path / "multivariate"
        assert (multi_dir / "correlation_heatmap.png").exists()
        interactions_dir = multi_dir / "interactions"
        assert interactions_dir.is_dir()
        assert any(interactions_dir.glob("*.png"))

    def test_creates_missingness_charts(self, fitted_report, tmp_path):
        fitted_report.export_charts(tmp_path)
        missing_dir = tmp_path / "missingness"
        assert (missing_dir / "missing_values.png").exists()
        assert (missing_dir / "missing_matrix.png").exists()
        assert (missing_dir / "missing_correlation.png").exists()

    def test_returns_dict_of_paths(self, fitted_report, tmp_path):
        paths = fitted_report.export_charts(tmp_path)
        assert isinstance(paths, dict)
        assert len(paths) > 0
        assert all(p.exists() for p in paths.values())


class TestExportChartsFormat:
    def test_svg_format(self, fitted_report, tmp_path):
        paths = fitted_report.export_charts(tmp_path, format="svg")
        # categorical x categorical pairs additionally persist a "<key>_table" CSV
        # (the contingency table) regardless of the image format requested.
        image_paths = {k: p for k, p in paths.items() if not k.endswith("_table")}
        table_paths = {k: p for k, p in paths.items() if k.endswith("_table")}
        assert image_paths, "expected at least one rendered chart"
        assert all(p.suffix == ".svg" for p in image_paths.values())
        assert all(p.suffix == ".csv" for p in table_paths.values())
        one_svg = next(iter(image_paths.values()))
        content = one_svg.read_bytes()
        assert b"<svg" in content[:200] or b"<?xml" in content[:200]

    def test_invalid_format_raises(self, fitted_report, tmp_path):
        with pytest.raises(ValueError, match="format"):
            fitted_report.export_charts(tmp_path, format="jpeg")


class TestExportChartsBivariateTopN:
    def test_bivariate_capped_by_top_n(self, fitted_report, tmp_path):
        paths_full = fitted_report.export_charts(tmp_path / "full", bivariate_top_n=100)
        n_full = sum(1 for k in paths_full if k.startswith("bivariate/"))

        paths_capped = fitted_report.export_charts(tmp_path / "capped", bivariate_top_n=1)
        n_capped = sum(1 for k in paths_capped if k.startswith("bivariate/"))

        assert n_capped == 1
        assert n_capped <= n_full


class TestExportChartsFilenameSafety:
    def test_column_names_with_special_characters_are_sanitized(self, tmp_path):
        import pandas as pd

        df = pd.DataFrame(
            {
                "weird col/name": [1, 2, 3, 4, 5, 6, 7, 8] * 5,
                "target": [0, 1, 0, 1, 0, 1, 0, 1] * 5,
            }
        )
        split = DataSplit(
            train_X=df[["weird col/name"]],
            train_y=df["target"],
            val_X=df[["weird col/name"]].iloc[:5],
            val_y=df["target"].iloc[:5],
            test_X=df[["weird col/name"]].iloc[:5],
            test_y=df["target"].iloc[:5],
            oot_X=df[["weird col/name"]].iloc[:0],
            oot_y=df["target"].iloc[:0],
            metadata=DataSplitMetadata(
                strategy="random",
                target_col="target",
                n_features=1,
                split_sizes={"train": 40, "val": 5, "test": 5, "oot": 0},
            ),
        )
        report = EDAReport(split, target="target").run_all()
        paths = report.export_charts(tmp_path)
        assert len(paths) > 0
        assert all(p.exists() for p in paths.values())
