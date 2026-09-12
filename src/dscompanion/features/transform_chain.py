"""Per-feature ordered transformation chains.

``ColumnRecipe``/``TransformStep`` are the human-readable, JSON-loggable
specification of an ordered transformation chain for one column (e.g.
``balance``: impute -> clip_lower -> clip_upper -> log1p -> bucket_quantile
-> target_encode). ``FeatureTransformChain`` compiles a mapping of these
recipes into real per-column ``sklearn.pipeline.Pipeline`` objects and
executes them — the fitted chain (opaque, pickled) stays separate from the
recipe specification (small, diffable JSON), so recipes can be compared
across experiment runs independently of how they were executed.

Does not use ``sklearn.compose.ColumnTransformer``: verified empirically
that every existing dscompanion transformer's ``get_feature_names_out()`` takes
no arguments, while sklearn's ``ColumnTransformer``/``Pipeline`` call it as
``get_feature_names_out(input_features)`` whenever pandas output or
``get_feature_names_out()`` itself is requested, raising ``TypeError``.
Each dscompanion transformer's own ``transform()`` already returns a correctly
named ``pd.DataFrame``, so per-column routing and concatenation are done
directly here instead.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

logger = logging.getLogger(__name__)

import pandas as pd
from pydantic import BaseModel, Field, field_validator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from dscompanion.features.registry import build_transformer, registered_transformer_names

__all__ = ["ColumnRecipe", "FeatureTransformChain", "TransformStep"]


class TransformStep(BaseModel):
    """One named transformation step within a ``ColumnRecipe``.

    Args:
        transformer (str): Registry key identifying which dscompanion transformer
            class to instantiate — see
            ``dscompanion.features.registry.registered_transformer_names()`` for
            the full list of valid values (e.g. ``"log"``, ``"clip_lower"``,
            ``"bucket_quantile"``).
        params (dict[str, Any]): Keyword arguments forwarded to the
            transformer's constructor, overriding its defaults. Defaults to
            an empty dict (use the transformer's own default parameters).

    Raises:
        ValueError: If ``transformer`` is not a registered name.
    """

    transformer: str
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("transformer")
    @classmethod
    def _validate_transformer_name(cls, value: str) -> str:
        valid = registered_transformer_names()
        if value not in valid:
            raise ValueError(f"Unknown transform step {value!r} — must be one of {valid}")
        return value


class ColumnRecipe(BaseModel):
    """An ordered, human-readable transformation recipe for one column.

    The unit meant to be logged as JSON and rendered in a model
    card — small and diffable between experiment runs, independent of the
    fitted (opaque, pickled) ``FeatureTransformChain`` that executes it.

    Args:
        column (str): Name of the column this recipe applies to.
        steps (list[TransformStep]): Ordered list of transformation steps,
            applied in sequence (step 1's output feeds step 2's input, etc.).
            An empty list means the column passes through unchanged.
        source (Literal["manual", "recommended"]): Whether this recipe was
            hand-built by a user or produced by a theory-driven
            recommendation engine. Defaults to ``"manual"``.
        rationale (str | None): Human-readable explanation of why these
            steps were chosen — most useful for a ``"recommended"`` recipe
            under audit/compliance review. Defaults to ``None``.
    """

    column: str
    steps: list[TransformStep] = Field(default_factory=list)
    source: Literal["manual", "recommended"] = "manual"
    rationale: str | None = None


class FeatureTransformChain(BaseEstimator, TransformerMixin):
    """Applies a distinct ordered transformation chain to each configured column.

    Compiles each column's ``ColumnRecipe`` into a real
    ``sklearn.pipeline.Pipeline`` of dscompanion transformer instances (resolved
    via the transformer registry). Columns with no recipe, or an empty-step
    recipe, pass through unchanged.

    Args:
        recipes (dict[str, ColumnRecipe]): Mapping from column name to its
            ordered transformation recipe.

    Attributes:
        column_pipelines_ (dict[str, Pipeline]): Fitted per-column
            ``sklearn.pipeline.Pipeline`` for every column with a non-empty
            recipe. Columns absent from this dict at transform time are
            passed through unchanged.
        output_columns_ (dict[str, list[str]]): Mapping from input column
            name to the list of output column names it produced during
            ``fit`` — 1 name for a 1:1 transform, more for a
            column-expanding one (e.g. ``onehot_encode``), always
            ``[column]`` for passthrough columns.
        column_order_ (list[str]): Column names in the order seen at
            ``fit`` time, used to assemble ``transform()``'s output in the
            same left-to-right order.
    """

    def __init__(self, recipes: dict[str, ColumnRecipe]) -> None:
        self.recipes = recipes

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureTransformChain":
        """Fits one ``sklearn.pipeline.Pipeline`` per column with a non-empty recipe.

        Args:
            X (pd.DataFrame): Training feature DataFrame.
            y (pd.Series | None): Target series, forwarded to every step's
                ``fit()`` — required by steps such as ``"target_encode"``/
                ``"woe_encode"``; ignored by steps that don't need it.

        Returns:
            FeatureTransformChain: The fitted instance (``self``), enabling
            method chaining.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``X`` has zero rows, or a recipe references a
                column not present in ``X``.
        """
        self._fit_columns(X, y)
        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fits every column's pipeline and returns its fitted training-time output.

        Explicitly overrides sklearn's default ``TransformerMixin.fit_transform()``
        (``fit().transform()`` composition). ``fit()`` already runs each column's
        ``pipeline.fit_transform()`` internally to learn output column names, but
        discards the resulting data block; a subsequent plain ``transform()`` call
        would re-run each step's ``.transform()``, which for steps such as
        ``"noise"`` (see ``NoiseInjector``) is a deliberate clean passthrough and
        would silently return unaugmented data. This method returns the actual
        fitted blocks instead, so train-time-only steps produce usable output.

        Args:
            X (pd.DataFrame): Training feature DataFrame.
            y (pd.Series | None): Target series, forwarded to every step's
                ``fit()`` — required by steps such as ``"target_encode"``/
                ``"woe_encode"``; ignored by steps that don't need it.

        Returns:
            pd.DataFrame: Concatenation of every column's fitted training-time
            block, in fit-time column order.

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``X`` has zero rows, or a recipe references a
                column not present in ``X``.
        """
        blocks_by_col = self._fit_columns(X, y)
        ordered_blocks = [blocks_by_col[col] for col in self.column_order_]
        return pd.concat(ordered_blocks, axis=1) if ordered_blocks else X.iloc[:, 0:0].copy()

    def _fit_columns(self, X: pd.DataFrame, y: pd.Series | None = None) -> dict[str, pd.DataFrame]:
        """Shared fit logic for ``fit()``/``fit_transform()`` — builds pipelines, returns blocks.

        Args:
            X (pd.DataFrame): Training feature DataFrame.
            y (pd.Series | None): Target series, forwarded to every step's ``fit()``.

        Returns:
            dict[str, pd.DataFrame]: Mapping from input column name to its
            fitted training-time output block (the column itself, unchanged,
            for passthrough columns).

        Raises:
            TypeError: If ``X`` is not a ``pd.DataFrame``.
            ValueError: If ``X`` has zero rows, or a recipe references a
                column not present in ``X``.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "FeatureTransformChain.fit() expects a pd.DataFrame, got %s" % type(X).__name__
            )
        if len(X) == 0:
            raise ValueError("FeatureTransformChain.fit() received an empty DataFrame (0 rows)")

        unknown_cols = [c for c in self.recipes if c not in X.columns]
        if unknown_cols:
            raise ValueError(f"ColumnRecipe(s) reference columns not present in X: {unknown_cols}")

        self.column_order_: list[str] = X.columns.tolist()
        self.column_pipelines_: dict[str, Pipeline] = {}
        self.output_columns_: dict[str, list[str]] = {}
        blocks: dict[str, pd.DataFrame] = {}

        for col in self.column_order_:
            recipe = self.recipes.get(col)
            if recipe is None or not recipe.steps:
                self.output_columns_[col] = [col]
                blocks[col] = X[[col]]
                continue

            steps = [
                (f"{i}_{step.transformer}", build_transformer(step.transformer, step.params))
                for i, step in enumerate(recipe.steps)
            ]
            pipeline = Pipeline(steps=steps)
            fitted_block = pipeline.fit_transform(X[[col]], y)
            self.column_pipelines_[col] = pipeline
            self.output_columns_[col] = list(fitted_block.columns)
            blocks[col] = fitted_block

        logger.info(
            "FeatureTransformChain fitted — %d columns with recipes, %d passthrough",
            len(self.column_pipelines_),
            len(self.column_order_) - len(self.column_pipelines_),
        )
        return blocks

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies every column's fitted pipeline (or passes it through unchanged).

        Args:
            X (pd.DataFrame): Feature DataFrame to transform. Columns
                absent from ``column_order_`` (unseen at fit time) are
                dropped from the output; columns present at fit time but
                missing here are skipped.

        Returns:
            pd.DataFrame: Concatenation of every column's transformed
            block, in fit-time column order. Column-expanding steps (e.g.
            ``onehot_encode``) contribute multiple output columns in that
            column's position.

        Raises:
            NotFittedError: If called before ``fit()``.
            TypeError: If ``X`` is not a ``pd.DataFrame``.
        """
        check_is_fitted(self, attributes=["column_pipelines_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "FeatureTransformChain.transform() expects a pd.DataFrame, got %s"
                % type(X).__name__
            )

        logger.debug("FeatureTransformChain.transform — rows=%d", len(X))

        blocks: list[pd.DataFrame] = []
        for col in self.column_order_:
            if col not in X.columns:
                continue
            if col in self.column_pipelines_:
                blocks.append(self.column_pipelines_[col].transform(X[[col]]))
            else:
                blocks.append(X[[col]])
        return pd.concat(blocks, axis=1) if blocks else X.iloc[:, 0:0].copy()

    def get_feature_names_out(self) -> list[str]:
        """Returns the ordered list of column names produced by ``transform``.

        Args:
            None

        Returns:
            list[str]: Output column names in fit-time column order,
            expanding in place for any column-expanding step.

        Raises:
            NotFittedError: If called before ``fit()``.
        """
        check_is_fitted(self, attributes=["output_columns_"])
        names: list[str] = []
        for col in self.column_order_:
            names.extend(self.output_columns_[col])
        return names
