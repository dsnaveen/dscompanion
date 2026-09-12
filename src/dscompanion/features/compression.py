"""PCATransformer: dimensionality reduction as a real fitted pipeline transformer."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_is_fitted

from dscompanion.config import settings

logger = logging.getLogger(__name__)

__all__ = ["PCATransformer"]


class PCATransformer(BaseEstimator, TransformerMixin):
    """Fits PCA on numeric columns and replaces them with principal components.

    Wraps sklearn's ``PCA`` as a DataFrame-in/DataFrame-out transformer,
    following this codebase's standard fit-on-train/apply-on-holdout pattern.
    ``dscompanion.eda.multivariate.MultivariateAnalyser`` already runs PCA, but
    only as an EDA diagnostic (scree plot, explained-variance table) — this
    is the fitted, transform-capable counterpart for actually reducing
    dimensionality in a model pipeline. Standalone — not wired into
    ``FeatureProcessingPipeline``, matching the precedent already set by
    ``AutoBinner``/``RareCategoryGrouper``: dimensionality reduction is an
    optional, model-specific decision, not a universal stage.

    Args:
        n_components (int | float | None): Number of components to keep. An
            ``int`` keeps exactly that many — capped (with an INFO log) to
            ``min(n_features, n_samples)`` if it exceeds what's available,
            since sklearn's ``PCA`` otherwise raises rather than capping
            itself, and ``settings.pca_default_components`` (10) commonly
            exceeds a small feature set. A ``float`` in ``(0, 1)`` keeps
            enough components to explain at least that fraction of variance
            (forwarded directly to sklearn's ``PCA``, no capping needed —
            it's naturally bounded by the max possible rank). When ``None``
            (default), uses ``settings.pca_default_components``.
        features (list[str] | None): Explicit numeric columns to reduce.
            When ``None`` (default), all numeric columns are used.
        prefix (str): Prefix for output component column names. Defaults to
            ``"pc"`` (columns named ``"pc1"``, ``"pc2"``, ...).

    Attributes:
        features_ (list[str]): Input columns the transform was fitted on —
            dropped from the output and replaced by principal components.
        explained_variance_ratio_ (np.ndarray): Fraction of variance
            explained by each retained component, populated after ``fit``.
    """

    def __init__(
        self,
        n_components: int | float | None = None,
        features: list[str] | None = None,
        prefix: str = "pc",
    ) -> None:
        self.n_components = n_components
        self.features = features
        self.prefix = prefix

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "PCATransformer":
        """Fits sklearn's ``PCA`` on the fitted numeric columns, storing the
        fitted estimator, component column names, and
        ``explained_variance_ratio_`` as side-effects.

        Missing values are filled with ``0`` for the purpose of fitting only
        — the same convention ``SmartScaler`` already uses — since PCA
        cannot handle ``NaN`` natively; run ``SmartImputer`` upstream for a
        real imputation strategy instead of relying on this fallback.

        Args:
            X (pd.DataFrame): Training feature DataFrame.
            y (pd.Series | None): Ignored. Present for sklearn pipeline
                compatibility.

        Returns:
            PCATransformer: The fitted instance (``self``), enabling method
            chaining.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``X`` has zero rows, or fewer than 2 numeric
                columns are available to reduce.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "PCATransformer.fit() expects a pd.DataFrame, got %s" % type(X).__name__
            )
        if len(X) == 0:
            raise ValueError("PCATransformer.fit() received an empty DataFrame (0 rows)")

        cols = (
            self.features
            if self.features is not None
            else X.select_dtypes(include="number").columns.tolist()
        )
        self.features_: list[str] = [c for c in cols if c in X.columns]
        if len(self.features_) < 2:
            raise ValueError(
                f"PCATransformer needs at least 2 numeric columns, got {len(self.features_)}"
            )

        n_components = (
            self.n_components if self.n_components is not None else settings.pca_default_components
        )
        max_components = min(len(self.features_), len(X))
        if isinstance(n_components, int) and n_components > max_components:
            logger.info(
                "PCATransformer: requested n_components=%d exceeds min(n_features, n_samples)"
                "=%d — capping to %d.",
                n_components,
                max_components,
                max_components,
            )
            n_components = max_components

        self._pca = PCA(n_components=n_components, random_state=settings.random_state)
        self._pca.fit(X[self.features_].fillna(0))
        self.explained_variance_ratio_ = self._pca.explained_variance_ratio_
        self._component_names: list[str] = [
            f"{self.prefix}{i + 1}" for i in range(self._pca.n_components_)
        ]

        logger.info(
            "PCATransformer fitted — %d input cols -> %d components, %.1f%% variance explained",
            len(self.features_),
            self._pca.n_components_,
            float(self.explained_variance_ratio_.sum()) * 100,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replaces the fitted numeric columns with principal components,
        dropping the originals; every other column is passed through
        unchanged. The original DataFrame is not modified.

        Args:
            X (pd.DataFrame): Feature DataFrame to transform. Must contain
                every column fitted on — unlike most transformers in this
                codebase, a missing fitted column can't be silently skipped:
                PCA is a linear combination of all of them, so dropping one
                would silently produce numerically wrong components rather
                than an incomplete-but-correct result.

        Returns:
            pd.DataFrame: A copy of ``X`` with the fitted numeric columns
            replaced by ``self._pca.n_components_`` principal-component
            columns (named via ``prefix``).

        Raises:
            NotFittedError: If called before ``fit()``.
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If any fitted column is missing from ``X``.
        """
        check_is_fitted(self, attributes=["_pca"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "PCATransformer.transform() expects a pd.DataFrame, got %s" % type(X).__name__
            )
        missing = [c for c in self.features_ if c not in X.columns]
        if missing:
            raise ValueError(f"PCATransformer.transform() missing fitted columns: {missing}")

        logger.debug("PCATransformer.transform — rows=%d", len(X))

        components = self._pca.transform(X[self.features_].fillna(0))
        out = X.drop(columns=self.features_)
        pc_df = pd.DataFrame(components, columns=self._component_names, index=X.index)
        return pd.concat([out, pc_df], axis=1)

    def get_feature_names_out(self) -> list[str]:
        """Returns the principal-component output column names.

        Args:
            None

        Returns:
            list[str]: Component column names, e.g. ``["pc1", "pc2", "pc3"]``.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["_pca"])
        return list(self._component_names)
