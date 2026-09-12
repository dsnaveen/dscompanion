"""Tests for dscompanion.explain — SHAPExplainer, LIMEExplainer, PDPAnalyser,
PermutationImportanceAnalyser."""

from __future__ import annotations

import numpy as np
import pytest

from dscompanion.models import ModelFactory


def _shap_available() -> bool:
    try:
        import shap  # noqa: F401

        return True
    except ImportError:
        return False


def _lime_available() -> bool:
    try:
        import lime  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def logistic_model(data_split):
    X = data_split.X_train.select_dtypes(include="number").fillna(0)
    y = data_split.y_train
    model = ModelFactory.build("classification", "logistic")
    model.fit(X, y)
    return model, X, y


# ---------------------------------------------------------------------------
# SHAPExplainer
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _shap_available(), reason="shap not installed")
class TestSHAPExplainer:
    def _small(self, X, n=100):
        return X.sample(n, random_state=42).reset_index(drop=True)

    def test_fit_completes(self, logistic_model):
        from dscompanion.explain import SHAPExplainer

        model, X, _ = logistic_model
        X_small = self._small(X)
        exp = SHAPExplainer(model, explainer_type="linear")
        exp.fit(X_small)
        assert exp.shap_values_.shape[0] == len(X_small)

    def test_mean_abs_shap_one_row_per_feature(self, logistic_model):
        from dscompanion.explain import SHAPExplainer

        model, X, _ = logistic_model
        X_small = self._small(X)
        exp = SHAPExplainer(model, explainer_type="linear")
        exp.fit(X_small)
        df = exp.mean_abs_shap()
        assert len(df) == X_small.shape[1]
        assert "feature" in df.columns
        assert "mean_abs_shap" in df.columns

    def test_waterfall_plot_returns_figure(self, logistic_model):
        import plotly.graph_objects as go

        from dscompanion.explain import SHAPExplainer

        model, X, _ = logistic_model
        X_small = self._small(X)
        exp = SHAPExplainer(model, explainer_type="linear")
        exp.fit(X_small)
        fig = exp.waterfall_plot(0)
        assert isinstance(fig, go.Figure)

    def test_summary_plot_returns_figure(self, logistic_model):
        import plotly.graph_objects as go

        from dscompanion.explain import SHAPExplainer

        model, X, _ = logistic_model
        X_small = self._small(X)
        exp = SHAPExplainer(model, explainer_type="linear")
        exp.fit(X_small)
        fig = exp.summary_plot()
        assert isinstance(fig, go.Figure)

    def test_dependence_plot_returns_figure(self, logistic_model):
        import plotly.graph_objects as go

        from dscompanion.explain import SHAPExplainer

        model, X, _ = logistic_model
        X_small = self._small(X)
        exp = SHAPExplainer(model, explainer_type="linear")
        exp.fit(X_small)
        first_feature = X_small.columns[0]
        fig = exp.dependence_plot(first_feature)
        assert isinstance(fig, go.Figure)

    def test_fit_raises_type_error_for_non_dataframe(self, logistic_model):
        from dscompanion.explain import SHAPExplainer

        model, _, _ = logistic_model
        exp = SHAPExplainer(model, explainer_type="linear")
        with pytest.raises(TypeError):
            exp.fit([[1, 2], [3, 4]])

    def test_mean_abs_shap_raises_before_fit(self, logistic_model):
        from dscompanion.explain import SHAPExplainer

        model, _, _ = logistic_model
        exp = SHAPExplainer(model, explainer_type="linear")
        with pytest.raises(Exception):
            exp.mean_abs_shap()

    def test_auto_explainer_type_is_detected(self, logistic_model):
        from dscompanion.explain import SHAPExplainer

        model, X, _ = logistic_model
        X_small = self._small(X)
        exp = SHAPExplainer(model, explainer_type="auto")
        exp.fit(X_small)
        assert exp.shap_values_.shape[0] == len(X_small)

    @pytest.fixture
    def xgb_model(self, data_split):
        X = data_split.X_train.select_dtypes(include="number").fillna(0)
        y = data_split.y_train
        X_val = data_split.X_val.select_dtypes(include="number").fillna(0)
        y_val = data_split.y_val
        model = ModelFactory.build("classification", "xgboost")
        model.fit(X, y, eval_set=[(X_val, y_val)])
        return model, X, y

    def test_fit_raises_value_error_for_empty_dataframe(self, logistic_model):
        import pandas as pd

        from dscompanion.explain import SHAPExplainer

        model, X, _ = logistic_model
        exp = SHAPExplainer(model, explainer_type="linear")
        with pytest.raises(ValueError, match="0 rows"):
            exp.fit(pd.DataFrame(columns=X.columns))

    def test_fit_with_tree_explainer(self, xgb_model):
        from dscompanion.explain import SHAPExplainer

        model, X, _ = xgb_model
        X_small = self._small(X)
        exp = SHAPExplainer(model, explainer_type="tree")
        exp.fit(X_small)
        assert exp.shap_values_.shape[0] == len(X_small)

    def test_waterfall_plot_raises_for_invalid_idx(self, logistic_model):
        from dscompanion.explain import SHAPExplainer

        model, X, _ = logistic_model
        X_small = self._small(X)
        exp = SHAPExplainer(model, explainer_type="linear")
        exp.fit(X_small)
        with pytest.raises(ValueError, match="out of range"):
            exp.waterfall_plot(99999)

    def test_dependence_plot_raises_for_unknown_feature(self, logistic_model):
        from dscompanion.explain import SHAPExplainer

        model, X, _ = logistic_model
        X_small = self._small(X)
        exp = SHAPExplainer(model, explainer_type="linear")
        exp.fit(X_small)
        with pytest.raises(ValueError, match="not found"):
            exp.dependence_plot("nonexistent_feature_xyz")

    def test_dependence_plot_with_interaction_feature(self, logistic_model):
        import plotly.graph_objects as go

        from dscompanion.explain import SHAPExplainer

        model, X, _ = logistic_model
        X_small = self._small(X)
        exp = SHAPExplainer(model, explainer_type="linear")
        exp.fit(X_small)
        first_feature = X_small.columns[0]
        second_feature = X_small.columns[1]
        fig = exp.dependence_plot(first_feature, interaction_feature=second_feature)
        assert isinstance(fig, go.Figure)


@pytest.mark.skipif(not _shap_available(), reason="shap not installed")
class TestBootstrapSHAPExplainer:
    @pytest.fixture
    def xgb_model(self, data_split):
        X = data_split.X_train.select_dtypes(include="number").fillna(0)
        y = data_split.y_train
        X_val = data_split.X_val.select_dtypes(include="number").fillna(0)
        y_val = data_split.y_val
        from dscompanion.models import ModelFactory

        model = ModelFactory.build("classification", "xgboost")
        model.fit(X, y, eval_set=[(X_val, y_val)])
        return model, X, y

    def test_fit_completes_and_sets_mean_shap(self, xgb_model):
        from dscompanion.explain.shap_explainer import BootstrapSHAPExplainer

        model, X, y = xgb_model
        bexp = BootstrapSHAPExplainer(model, n_batches=2, batch_size=50, background_size=30)
        bexp.fit(X.head(200), y.head(200))
        assert bexp.mean_shap_values_ is not None
        # mean_shap_values_ has shape (batch_size, n_features)
        assert bexp.mean_shap_values_.shape[1] == X.shape[1]

    def test_get_feature_importance_returns_sorted_dataframe(self, xgb_model):
        from dscompanion.explain.shap_explainer import BootstrapSHAPExplainer

        model, X, y = xgb_model
        bexp = BootstrapSHAPExplainer(model, n_batches=2, batch_size=50, background_size=30)
        bexp.fit(X.head(200), y.head(200))
        df = bexp.get_feature_importance()
        assert "importance" in df.columns
        assert "rank" in df.columns
        assert len(df) == X.shape[1]

    def test_beeswarm_plot_returns_figure(self, xgb_model):
        import plotly.graph_objects as go

        from dscompanion.explain.shap_explainer import BootstrapSHAPExplainer

        model, X, y = xgb_model
        bexp = BootstrapSHAPExplainer(model, n_batches=2, batch_size=50, background_size=30)
        bexp.fit(X.head(200), y.head(200))
        fig = bexp.beeswarm_plot(max_display=5)
        assert isinstance(fig, go.Figure)

    def test_importance_plot_returns_figure(self, xgb_model):
        import plotly.graph_objects as go

        from dscompanion.explain.shap_explainer import BootstrapSHAPExplainer

        model, X, y = xgb_model
        bexp = BootstrapSHAPExplainer(model, n_batches=2, batch_size=50, background_size=30)
        bexp.fit(X.head(200), y.head(200))
        fig = bexp.importance_plot(max_display=5)
        assert isinstance(fig, go.Figure)

    def test_get_summary_stats_returns_dict(self, xgb_model):
        from dscompanion.explain.shap_explainer import BootstrapSHAPExplainer

        model, X, y = xgb_model
        bexp = BootstrapSHAPExplainer(model, n_batches=2, batch_size=50, background_size=30)
        bexp.fit(X.head(200), y.head(200))
        stats = bexp.get_summary_stats()
        assert isinstance(stats, dict)
        assert "top_5_features" in stats

    def test_fit_raises_type_error_for_non_dataframe(self, xgb_model):
        from dscompanion.explain.shap_explainer import BootstrapSHAPExplainer

        model, _, _ = xgb_model
        bexp = BootstrapSHAPExplainer(model, n_batches=2, batch_size=50)
        with pytest.raises(TypeError):
            bexp.fit([[1, 2], [3, 4]])

    def test_fit_raises_value_error_for_empty_dataframe(self, xgb_model):
        import pandas as pd

        from dscompanion.explain.shap_explainer import BootstrapSHAPExplainer

        model, X, _ = xgb_model
        bexp = BootstrapSHAPExplainer(model, n_batches=2, batch_size=50)
        with pytest.raises(ValueError):
            bexp.fit(pd.DataFrame(columns=X.columns))

    def test_fit_without_y_uses_random_sampling(self, xgb_model):
        from dscompanion.explain.shap_explainer import BootstrapSHAPExplainer

        model, X, _ = xgb_model
        bexp = BootstrapSHAPExplainer(model, n_batches=2, batch_size=50, background_size=30)
        bexp.fit(X.head(200))  # no y → random sampling path
        assert hasattr(bexp, "shap_values_")

    def test_fit_with_linear_model_uses_linear_explainer(self, data_split):
        from dscompanion.explain.shap_explainer import BootstrapSHAPExplainer

        X = data_split.X_train.select_dtypes(include="number").fillna(0)
        y = data_split.y_train
        model = ModelFactory.build("classification", "logistic")
        model.fit(X, y)
        bexp = BootstrapSHAPExplainer(model, n_batches=2, batch_size=50, background_size=30)
        bexp.fit(X.head(200), y.head(200))
        assert hasattr(bexp, "shap_values_")

    def test_dependence_plot_returns_figure(self, xgb_model):
        import plotly.graph_objects as go

        from dscompanion.explain.shap_explainer import BootstrapSHAPExplainer

        model, X, y = xgb_model
        bexp = BootstrapSHAPExplainer(model, n_batches=2, batch_size=50, background_size=30)
        bexp.fit(X.head(200), y.head(200))
        fig = bexp.dependence_plot(X.columns[0])
        assert isinstance(fig, go.Figure)

    def test_dependence_plot_raises_before_fit(self, xgb_model):
        from sklearn.exceptions import NotFittedError

        from dscompanion.explain.shap_explainer import BootstrapSHAPExplainer

        model, _, _ = xgb_model
        bexp = BootstrapSHAPExplainer(model, n_batches=2, batch_size=50)
        with pytest.raises(NotFittedError):
            bexp.dependence_plot("some_feature")

    def test_dependence_plot_raises_for_unknown_feature(self, xgb_model):
        from dscompanion.explain.shap_explainer import BootstrapSHAPExplainer

        model, X, y = xgb_model
        bexp = BootstrapSHAPExplainer(model, n_batches=2, batch_size=50, background_size=30)
        bexp.fit(X.head(200), y.head(200))
        with pytest.raises(ValueError, match="not found"):
            bexp.dependence_plot("nonexistent_feature_xyz")

    def test_export_shap_values_writes_parquet(self, xgb_model, tmp_path):
        import pandas as pd

        from dscompanion.explain.shap_explainer import BootstrapSHAPExplainer

        model, X, y = xgb_model
        bexp = BootstrapSHAPExplainer(model, n_batches=2, batch_size=50, background_size=30)
        bexp.fit(X.head(200), y.head(200))
        out_path = tmp_path / "shap_export.parquet"
        bexp.export_shap_values(out_path)
        assert out_path.exists()
        df = pd.read_parquet(out_path)
        assert "batch_id" in df.columns
        assert "expected_value" in df.columns

    def test_save_and_load_round_trip(self, xgb_model, tmp_path):
        from dscompanion.explain.shap_explainer import BootstrapSHAPExplainer

        model, X, y = xgb_model
        bexp = BootstrapSHAPExplainer(model, n_batches=2, batch_size=50, background_size=30)
        bexp.fit(X.head(200), y.head(200))
        save_path = tmp_path / "bexp.joblib"
        bexp.save(save_path)
        loaded = BootstrapSHAPExplainer.load(save_path)
        np.testing.assert_array_almost_equal(loaded.mean_shap_values_, bexp.mean_shap_values_)


@pytest.mark.skipif(not _lime_available(), reason="lime not installed")
class TestLIMEExplainer:
    def test_explain_instance_returns_figure(self, logistic_model):
        import plotly.graph_objects as go

        from dscompanion.explain import LIMEExplainer

        model, X, _ = logistic_model
        lime_exp = LIMEExplainer(model, training_data=X, mode="classification", n_samples=500)
        fig = lime_exp.explain_instance(X.iloc[0])
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


# ---------------------------------------------------------------------------
# PDPAnalyser
# ---------------------------------------------------------------------------


class TestPDPAnalyser:
    """Tests for PDPAnalyser (B.9 — full implementation)."""

    @pytest.fixture
    def pdp_fixture(self, data_split):
        """200-row numeric-only dataset + fitted logistic model."""
        X_full = data_split.X_train.select_dtypes(include="number").fillna(0)
        X = X_full.head(200).reset_index(drop=True)
        y = data_split.y_train.iloc[: len(X)].reset_index(drop=True)
        model = ModelFactory.build("classification", "logistic")
        model.fit(X, y)
        return model, X, y

    def test_fit_completes(self, pdp_fixture):
        from dscompanion.explain import PDPAnalyser

        model, X, _ = pdp_fixture
        features = list(X.columns[:2])
        pdp = PDPAnalyser(model, grid_resolution=5)
        pdp.fit(X, features)
        assert hasattr(pdp, "pdp_results_")
        assert set(features) == set(pdp.pdp_results_.keys())

    def test_fit_type_error_x_not_dataframe(self, pdp_fixture):
        from dscompanion.explain import PDPAnalyser

        model, X, _ = pdp_fixture
        pdp = PDPAnalyser(model, grid_resolution=5)
        with pytest.raises(TypeError, match="pd.DataFrame"):
            pdp.fit(X.values, list(X.columns[:1]))

    def test_fit_value_error_empty_dataframe(self, pdp_fixture):
        from dscompanion.explain import PDPAnalyser

        model, X, _ = pdp_fixture
        pdp = PDPAnalyser(model, grid_resolution=5)
        with pytest.raises(ValueError, match="0 rows"):
            pdp.fit(X.iloc[:0], list(X.columns[:1]))

    def test_fit_value_error_missing_feature(self, pdp_fixture):
        from dscompanion.explain import PDPAnalyser

        model, X, _ = pdp_fixture
        pdp = PDPAnalyser(model, grid_resolution=5)
        with pytest.raises(ValueError, match="not found"):
            pdp.fit(X, ["nonexistent_col_xyz"])

    def test_pdp_results_have_matching_grid_and_average_lengths(self, pdp_fixture):
        from dscompanion.explain import PDPAnalyser

        model, X, _ = pdp_fixture
        features = list(X.columns[:2])
        pdp = PDPAnalyser(model, grid_resolution=5)
        pdp.fit(X, features)
        for feat in features:
            res = pdp.pdp_results_[feat]
            assert "grid_values" in res
            assert "average" in res
            assert len(res["grid_values"]) == len(res["average"])
            assert len(res["grid_values"]) == 5

    def test_plot_returns_figure(self, pdp_fixture):
        import plotly.graph_objects as go

        from dscompanion.explain import PDPAnalyser

        model, X, _ = pdp_fixture
        feat = list(X.columns)[0]
        pdp = PDPAnalyser(model, grid_resolution=5)
        pdp.fit(X, [feat])
        fig = pdp.plot(feat)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_plot_raises_before_fit(self, pdp_fixture):
        from sklearn.exceptions import NotFittedError

        from dscompanion.explain import PDPAnalyser

        model, _, _ = pdp_fixture
        pdp = PDPAnalyser(model, grid_resolution=5)
        with pytest.raises(NotFittedError):
            pdp.plot("f1")

    def test_plot_raises_for_unknown_feature(self, pdp_fixture):
        from dscompanion.explain import PDPAnalyser

        model, X, _ = pdp_fixture
        feat = list(X.columns)[0]
        pdp = PDPAnalyser(model, grid_resolution=5)
        pdp.fit(X, [feat])
        with pytest.raises(ValueError, match="not found"):
            pdp.plot("nonexistent_col_xyz")

    def test_discover_thresholds_structure(self, pdp_fixture):
        from dscompanion.explain import PDPAnalyser

        model, X, _ = pdp_fixture
        features = list(X.columns[:2])
        pdp = PDPAnalyser(model, grid_resolution=5)
        pdp.fit(X, features)
        result = pdp.discover_thresholds()
        assert set(result.keys()) == set(features)
        for info in result.values():
            assert "thresholds" in info
            assert "baseline" in info
            assert "peak_prob" in info
            assert "range_of_effect" in info
            assert isinstance(info["thresholds"], list)

    def test_discover_thresholds_raises_before_fit(self, pdp_fixture):
        from sklearn.exceptions import NotFittedError

        from dscompanion.explain import PDPAnalyser

        model, _, _ = pdp_fixture
        pdp = PDPAnalyser(model, grid_resolution=5)
        with pytest.raises(NotFittedError):
            pdp.discover_thresholds()

    def test_discover_thresholds_custom_params(self, pdp_fixture):
        from dscompanion.explain import PDPAnalyser

        model, X, _ = pdp_fixture
        feat = list(X.columns)[0]
        pdp = PDPAnalyser(model, grid_resolution=10)
        pdp.fit(X, [feat])
        # very low min_slope should flag more thresholds than very high min_slope
        result_low = pdp.discover_thresholds(min_slope=0.0, min_prob_increase=0.0)
        result_high = pdp.discover_thresholds(min_slope=999.0, min_prob_increase=999.0)
        assert len(result_low[feat]["thresholds"]) >= len(result_high[feat]["thresholds"])

    def test_summary_table_has_expected_columns(self, pdp_fixture):
        from dscompanion.explain import PDPAnalyser

        model, X, _ = pdp_fixture
        features = list(X.columns[:2])
        pdp = PDPAnalyser(model, grid_resolution=5)
        pdp.fit(X, features)
        tbl = pdp.summary_table()
        for col in [
            "feature",
            "n_grid_points",
            "baseline_prob",
            "peak_prob",
            "trough_prob",
            "range_of_effect",
        ]:
            assert col in tbl.columns
        assert len(tbl) == len(features)

    def test_summary_table_raises_before_fit(self, pdp_fixture):
        from sklearn.exceptions import NotFittedError

        from dscompanion.explain import PDPAnalyser

        model, _, _ = pdp_fixture
        pdp = PDPAnalyser(model, grid_resolution=5)
        with pytest.raises(NotFittedError):
            pdp.summary_table()

    def test_summary_table_empty_schema_when_no_features(self, pdp_fixture):
        from dscompanion.explain import PDPAnalyser

        model, X, _ = pdp_fixture
        pdp = PDPAnalyser(model, grid_resolution=5)
        pdp.fit(X, [])
        tbl = pdp.summary_table()
        assert len(tbl) == 0
        assert "feature" in tbl.columns

    def test_pickle_round_trip(self, pdp_fixture, tmp_path):
        import joblib

        from dscompanion.explain import PDPAnalyser

        model, X, _ = pdp_fixture
        feat = list(X.columns)[0]
        pdp = PDPAnalyser(model, grid_resolution=5)
        pdp.fit(X, [feat])
        path = tmp_path / "pdp.joblib"
        joblib.dump(pdp, path)
        loaded = joblib.load(path)
        np.testing.assert_array_almost_equal(
            loaded.pdp_results_[feat]["average"],
            pdp.pdp_results_[feat]["average"],
        )

    def test_get_params_returns_constructor_args(self, pdp_fixture):
        from dscompanion.explain import PDPAnalyser

        model, _, _ = pdp_fixture
        pdp = PDPAnalyser(model, grid_resolution=7)
        params = pdp.get_params()
        assert params["grid_resolution"] == 7
        assert params["model"] is model


# ---------------------------------------------------------------------------
# PermutationImportanceAnalyser
# ---------------------------------------------------------------------------


class TestPermutationImportanceAnalyser:
    """Tests for PermutationImportanceAnalyser (PI.1 — crash-safe SHAP alternative)."""

    @pytest.fixture
    def perm_fixture(self, data_split):
        """200-row numeric-only dataset + fitted logistic model."""
        X_full = data_split.X_train.select_dtypes(include="number").fillna(0)
        X = X_full.head(200).reset_index(drop=True)
        y = data_split.y_train.iloc[: len(X)].reset_index(drop=True)
        model = ModelFactory.build("classification", "logistic")
        model.fit(X, y)
        return model, X, y

    def test_fit_completes(self, perm_fixture):
        from dscompanion.explain import PermutationImportanceAnalyser

        model, X, y = perm_fixture
        analyser = PermutationImportanceAnalyser(model, scoring="roc_auc", n_repeats=3)
        analyser.fit(X, y)
        assert hasattr(analyser, "importances_")

    def test_importance_table_one_row_per_feature(self, perm_fixture):
        from dscompanion.explain import PermutationImportanceAnalyser

        model, X, y = perm_fixture
        analyser = PermutationImportanceAnalyser(model, scoring="roc_auc", n_repeats=3)
        analyser.fit(X, y)
        df = analyser.importance_table()
        assert len(df) == X.shape[1]
        assert list(df.columns) == ["feature", "importance_mean", "importance_std", "rank"]
        assert df["rank"].tolist() == sorted(df["rank"].tolist())

    def test_importance_table_sorted_descending_by_mean(self, perm_fixture):
        from dscompanion.explain import PermutationImportanceAnalyser

        model, X, y = perm_fixture
        analyser = PermutationImportanceAnalyser(model, scoring="roc_auc", n_repeats=3)
        analyser.fit(X, y)
        df = analyser.importance_table()
        means = df["importance_mean"].tolist()
        assert means == sorted(means, reverse=True)

    def test_summary_plot_returns_figure(self, perm_fixture):
        import plotly.graph_objects as go

        from dscompanion.explain import PermutationImportanceAnalyser

        model, X, y = perm_fixture
        analyser = PermutationImportanceAnalyser(model, scoring="roc_auc", n_repeats=3, top_n=5)
        analyser.fit(X, y)
        fig = analyser.summary_plot()
        assert isinstance(fig, go.Figure)

    def test_fit_raises_type_error_for_non_dataframe(self, perm_fixture):
        from dscompanion.explain import PermutationImportanceAnalyser

        model, X, y = perm_fixture
        analyser = PermutationImportanceAnalyser(model, scoring="roc_auc")
        with pytest.raises(TypeError, match="pd.DataFrame"):
            analyser.fit(X.values, y)

    def test_fit_raises_value_error_for_empty_dataframe(self, perm_fixture):
        from dscompanion.explain import PermutationImportanceAnalyser

        model, X, y = perm_fixture
        analyser = PermutationImportanceAnalyser(model, scoring="roc_auc")
        with pytest.raises(ValueError, match="0 rows"):
            analyser.fit(X.iloc[:0], y.iloc[:0])

    def test_importance_table_raises_before_fit(self, perm_fixture):
        from dscompanion.explain import PermutationImportanceAnalyser

        model, _, _ = perm_fixture
        analyser = PermutationImportanceAnalyser(model, scoring="roc_auc")
        with pytest.raises(Exception):
            analyser.importance_table()

    def test_pickle_round_trip(self, perm_fixture, tmp_path):
        import joblib

        from dscompanion.explain import PermutationImportanceAnalyser

        model, X, y = perm_fixture
        analyser = PermutationImportanceAnalyser(model, scoring="roc_auc", n_repeats=3)
        analyser.fit(X, y)
        path = tmp_path / "perm.joblib"
        joblib.dump(analyser, path)
        loaded = joblib.load(path)
        pd_testing = __import__("pandas.testing", fromlist=["assert_frame_equal"])
        pd_testing.assert_frame_equal(loaded.importance_table(), analyser.importance_table())

    def test_get_params_returns_constructor_args(self, perm_fixture):
        from dscompanion.explain import PermutationImportanceAnalyser

        model, _, _ = perm_fixture
        analyser = PermutationImportanceAnalyser(model, scoring="roc_auc", n_repeats=7)
        params = analyser.get_params()
        assert params["n_repeats"] == 7
        assert params["model"] is model
