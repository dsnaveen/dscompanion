"""LIMEExplainer: LIME-based local model explanations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

from dscompanion.config import settings

if TYPE_CHECKING:
    import plotly.graph_objects as go

logger = logging.getLogger(__name__)

__all__ = ["LIMEExplainer"]


class LIMEExplainer:
    """Generates local, instance-level feature-importance explanations using
    the LIME (Local Interpretable Model-agnostic Explanations) algorithm.

    Internally wraps ``lime.lime_tabular.LimeTabularExplainer`` and lazily
    builds the explainer on first use.  Supports both classification (returns
    a bar chart of signed LIME weights) and regression modes.

    Args:
        model (Any): Fitted ``BaseDSCompanionModel`` instance.  Its ``predict_proba``
            method is used in classification mode; ``predict`` is used in
            regression mode.
        training_data (pd.DataFrame): Full training feature matrix used to
            construct the LIME background distribution (feature statistics and
            discretisation boundaries).
        mode (str): Explanation mode.  ``"classification"`` (default) returns
            signed weights for the positive class; ``"regression"`` returns
            weights for the raw prediction.
        n_features (int): Maximum number of top features to include in each
            explanation.  Defaults to 10.
        n_samples (int): Number of perturbed neighbourhood samples generated
            per LIME call.  Higher values improve stability at the cost of
            latency.  Defaults to 5000.
    """

    def __init__(
        self,
        model: Any,  # Any: fitted BaseDSCompanionModel — avoids circular import
        training_data: pd.DataFrame,
        mode: str = "classification",
        n_features: int = 10,
        n_samples: int = 5000,
    ) -> None:
        self.model = model
        self.training_data = training_data
        self.mode = mode
        self.n_features = n_features
        self.n_samples = n_samples
        self._explainer = None

    def _get_explainer(
        self,
    ) -> Any:  # Any: LimeTabularExplainer — avoids hard lime dep at module level
        if self._explainer is not None:
            return self._explainer
        try:
            import lime.lime_tabular as lt
        except ImportError as exc:
            raise ImportError("lime is required. pip install lime") from exc

        self._explainer = lt.LimeTabularExplainer(
            training_data=self.training_data.values,
            feature_names=list(self.training_data.columns),
            mode=self.mode,
            random_state=settings.random_state,
        )
        return self._explainer

    def explain_instance(self, row: pd.Series | pd.DataFrame) -> "go.Figure":
        """Generate a horizontal bar chart of LIME feature weights for a
        single observation, with positive weights shown in crimson and
        negative weights in steelblue.

        Lazily initialises the underlying ``LimeTabularExplainer`` on first
        call.  If ``row`` is a single-row DataFrame it is automatically
        squeezed to a Series before being passed to LIME.

        Args:
            row (pd.Series or pd.DataFrame): The observation to explain.
                Must be a single row — a ``pd.Series`` of length
                ``n_features`` or a ``pd.DataFrame`` with exactly one row.
                Column order must match ``training_data``.

        Returns:
            plotly.graph_objects.Figure: A horizontal bar chart with one bar
            per feature (up to ``n_features``), x-axis showing LIME weight,
            y-axis showing feature condition strings.

        Raises:
            ImportError: If ``plotly`` is not installed.
        """
        from dscompanion.utils.plotting import apply_dscompanion_theme

        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError("plotly is required.") from exc

        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        explainer = self._get_explainer()
        _cols = list(self.training_data.columns)
        _raw_fn = self.model.predict_proba if self.mode == "classification" else self.model.predict

        def predict_fn(X_arr):
            return _raw_fn(pd.DataFrame(X_arr, columns=_cols))

        exp = explainer.explain_instance(
            row.values,
            predict_fn,
            num_features=self.n_features,
            num_samples=self.n_samples,
        )
        local = exp.as_list()
        features = [f for f, _ in local]
        weights = [w for _, w in local]
        colors = ["crimson" if w > 0 else "steelblue" for w in weights]

        fig = go.Figure(
            go.Bar(
                x=weights,
                y=features,
                orientation="h",
                marker_color=colors,
            )
        )
        fig.update_layout(
            title="LIME Local Explanation",
            xaxis_title="LIME weight",
            yaxis=dict(autorange="reversed"),
        )
        apply_dscompanion_theme(fig)
        return fig

    def explain_batch(self, X: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """Compute LIME explanations for every row in ``X`` and return the
        results in long (tidy) format, with one row per feature per
        observation.

        Rows that fail to produce an explanation (e.g. due to a numerical
        error inside LIME) are silently skipped and a DEBUG message is
        logged, so the returned DataFrame may contain fewer than
        ``len(X) * top_n`` rows.

        Args:
            X (pd.DataFrame): Feature matrix of shape ``(n_samples, n_features)``
                to explain.  Column names must match ``training_data``.
            top_n (int): Number of highest-weight features to include per
                observation.  Defaults to 5.

        Returns:
            pd.DataFrame: Tidy DataFrame with columns ``idx`` (int, row
            position in ``X``), ``feature`` (str, LIME condition string),
            and ``lime_weight`` (float, signed LIME coefficient).  Returns
            an empty DataFrame with those three columns when no rows can be
            explained.
        """
        explainer = self._get_explainer()
        _cols = list(self.training_data.columns)
        _raw_fn = self.model.predict_proba if self.mode == "classification" else self.model.predict

        def predict_fn(X_arr):
            return _raw_fn(pd.DataFrame(X_arr, columns=_cols))

        rows = []
        for i, (_, row) in enumerate(X.iterrows()):
            try:
                exp = explainer.explain_instance(
                    row.values,
                    predict_fn,
                    num_features=top_n,
                    num_samples=self.n_samples,
                )
                for feat, weight in exp.as_list():
                    rows.append({"idx": i, "feature": feat, "lime_weight": weight})
            except Exception as exc:
                logger.debug("LIME explain_batch skipped row %d: %s", i, exc)
        return pd.DataFrame(rows)
