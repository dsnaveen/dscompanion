"""ScoringPipeline: bundles a trained dscompanion model with everything needed to
score new, unseen data.

Design spec: notes/superpowers/specs/2026-09-16-scoring-pipeline-design.md
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from dscompanion.config import settings
from dscompanion.utils.metrics import feature_csi_table, psi_score

logger = logging.getLogger(__name__)

__all__ = ["ScoringPipeline"]

_BUNDLE_SCHEMA_VERSION = 2


def _dtype_bucket(series: pd.Series) -> str:
    """Classify a Series' dtype into one of three coarse buckets for schema validation.

    Exact dtype (e.g. ``int64`` vs ``int32``) is deliberately not preserved —
    pandas/numpy upcast automatically during ``.transform()``; only a genuine
    kind change (e.g. a column that was numeric at training arrives as
    strings) should fail scoring-time validation.

    Args:
        series (pd.Series): Column to classify.

    Returns:
        str: One of ``"numeric"``, ``"datetime"``, or ``"categorical"``
        (the fallback bucket for anything else, e.g. object/string/bool).
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


class ScoringPipeline:
    """Bundles a trained dscompanion model with its fitted preprocessing chain so it
    can be applied to new, unseen raw data in a later process.

    Never fits anything itself — every attribute is an already-fitted object
    produced by ``PipelineRunner.run()`` (see ``from_run``). ``predict()``
    replays the exact same ``feature_pipeline``/``selection_pipeline``
    ``.transform()`` calls training used internally, so scoring can never
    diverge from what training actually learned. Stateless at call time
    (``predict``/``predict_one``/``compute_drift`` never mutate ``self``),
    so a single loaded instance is safe to reuse across many calls — a
    batch job scoring one large DataFrame, or a long-lived process (e.g. a
    future API service) scoring many small requests over its lifetime.

    Attributes:
        feature_pipeline: Fitted ``FeatureProcessingPipeline`` (or
            ``FeatureTransformChain``) from the training run.
        selection_pipeline: Fitted ``FeatureSelectionPipeline`` from the
            training run.
        model: Fitted ``BaseDSCompanionModel`` from the training run.
        calibrator: Fitted ``Calibrator``, or ``None`` when calibration
            wasn't applicable (regression/clustering) or failed during
            training.
        schema_ (dict[str, str]): ``{raw column name: dtype bucket}`` for
            every column ``feature_pipeline`` was fit on — see
            ``_dtype_bucket``. Required at scoring time; see ``predict``.
        target_col (str): Name of the target column used during training.
            Never expected in scoring input.
        task (str): ``"classification"``, ``"regression"``, or
            ``"clustering"``.
        id_column_candidates (list[str]): Raw columns excluded from
            features at training time (``data.ignore_columns`` plus
            ``data.date_column`` if set) — informational only, surfaced so
            calling code can discover what id-like columns were available;
            never a constraint on what ``id_columns`` a caller may pass to
            ``predict``.
        psi_reference_ (pd.Series | None): Training-set prediction score
            distribution (classification: positive-class probability,
            calibrated if a calibrator is present; regression: predicted
            value), used as the baseline for ``compute_drift``. ``None``
            for clustering (no comparable score distribution) or if it
            could not be computed during training.
        feature_reference_ (dict[str, dict] | None): Per-column frozen
            reference distribution (bin edges + proportions for numeric
            columns, category → proportion map for categorical columns)
            computed once from the raw training data via
            ``freeze_feature_reference`` — used as the baseline for
            ``compute_feature_drift``. Does NOT contain any raw training
            rows (privacy/size — a portable bundle should never carry
            actual customer feature values). ``None`` for
            ``task="clustering"`` or if it could not be computed during
            training.
        bundle_schema_version (int): Versions this class's on-disk shape,
            independent of ``dscompanion``'s package version — an
            incompatible future redesign bumps this, and ``load`` rejects
            a mismatched bundle with a clear error instead of silently
            misbehaving.
        dscompanion_version (str): ``dscompanion`` package version at save
            time. Diagnostic only.
    """

    def __init__(
        self,
        *,
        feature_pipeline: Any,
        selection_pipeline: Any,
        model: Any,
        calibrator: Any | None,
        schema: dict[str, str],
        target_col: str,
        task: str,
        id_column_candidates: list[str],
        psi_reference: pd.Series | None,
        feature_reference: dict[str, dict] | None = None,
        bundle_schema_version: int = _BUNDLE_SCHEMA_VERSION,
        dscompanion_version: str = "",
    ) -> None:
        self.feature_pipeline = feature_pipeline
        self.selection_pipeline = selection_pipeline
        self.model = model
        self.calibrator = calibrator
        self.schema_ = schema
        self.target_col = target_col
        self.task = task
        self.id_column_candidates = list(id_column_candidates)
        self.psi_reference_ = psi_reference
        self.feature_reference_ = feature_reference
        self.bundle_schema_version = bundle_schema_version
        self.dscompanion_version = dscompanion_version

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_run(
        cls,
        *,
        feature_pipeline: Any,
        selection_pipeline: Any,
        model: Any,
        calibrator: Any | None,
        schema_df: pd.DataFrame,
        target_col: str,
        task: str,
        id_column_candidates: list[str],
        psi_reference: pd.Series | None,
        feature_reference: dict[str, dict] | None = None,
    ) -> "ScoringPipeline":
        """Build a ``ScoringPipeline`` from a completed ``PipelineRunner`` run's artefacts.

        Args:
            feature_pipeline (Any): Fitted ``FeatureProcessingPipeline`` (or
                ``FeatureTransformChain``).
            selection_pipeline (Any): Fitted ``FeatureSelectionPipeline``.
            model (Any): Fitted ``BaseDSCompanionModel``.
            calibrator (Any | None): Fitted ``Calibrator``, or ``None``.
            schema_df (pd.DataFrame): The raw (pre-``feature_pipeline``)
                feature matrix ``feature_pipeline`` was fit on — its columns
                and dtypes become ``schema_``. Typically the training
                split's ``train_X`` immediately before feature processing.
            target_col (str): Training target column name.
            task (str): ``"classification"``, ``"regression"``, or
                ``"clustering"``.
            id_column_candidates (list[str]): Raw columns excluded from
                features at training time — see the class docstring.
            psi_reference (pd.Series | None): Training-set prediction score
                distribution for ``compute_drift``, or ``None``.
            feature_reference (dict[str, dict] | None): Frozen per-column
                reference distribution for ``compute_feature_drift``
                (from ``freeze_feature_reference``), or ``None``.

        Returns:
            ScoringPipeline: A new, ready-to-save instance.
        """
        import dscompanion

        schema = {col: _dtype_bucket(schema_df[col]) for col in schema_df.columns}
        return cls(
            feature_pipeline=feature_pipeline,
            selection_pipeline=selection_pipeline,
            model=model,
            calibrator=calibrator,
            schema=schema,
            target_col=target_col,
            task=task,
            id_column_candidates=id_column_candidates,
            psi_reference=psi_reference,
            feature_reference=feature_reference,
            dscompanion_version=getattr(dscompanion, "__version__", ""),
        )

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        """Serialize this ``ScoringPipeline`` to a single joblib file.

        Nested fitted objects (``feature_pipeline``, ``selection_pipeline``,
        ``model``, ``calibrator``) serialize automatically as part of the
        same pickle graph — no component needs its own save call.

        Args:
            path (str | Path): Destination file path. Parent directory must
                already exist.

        Returns:
            Path: The resolved path written to.
        """
        path = Path(path)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ScoringPipeline":
        """Deserialize a ``ScoringPipeline`` previously written by ``save``.

        Args:
            path (str | Path): Path to a joblib file written by ``save``.

        Returns:
            ScoringPipeline: The restored instance, ready for ``predict``/
            ``predict_one``/``compute_drift``.

        Raises:
            RuntimeError: If the bundle's ``bundle_schema_version`` doesn't
                match this ``dscompanion`` version's expected value — fails
                here, at load time, rather than silently misbehaving inside
                a later ``predict`` call.
        """
        obj = joblib.load(path)
        if obj.bundle_schema_version != _BUNDLE_SCHEMA_VERSION:
            raise RuntimeError(
                f"ScoringPipeline bundle at {path!r} has bundle_schema_version="
                f"{obj.bundle_schema_version}, but this dscompanion version expects "
                f"{_BUNDLE_SCHEMA_VERSION}. Re-train and re-save with the current "
                "dscompanion version, or install the dscompanion version that produced "
                "this bundle."
            )
        return obj

    # ── Scoring ──────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame, id_columns: list[str] | None = None) -> pd.DataFrame:
        """Score new, raw data and return predictions.

        Validates ``df``'s schema against ``schema_``, transforms it through
        the exact fitted ``feature_pipeline``/``selection_pipeline`` chain
        training used, then scores with ``model`` (and ``calibrator`` when
        present, for classification).

        Args:
            df (pd.DataFrame): Raw, unseen data — must contain every column
                in ``schema_`` (extra columns are tolerated and ignored) and
                must NOT need to contain ``target_col``.
            id_columns (list[str] | None): Raw columns from ``df`` to carry
                through to the output unchanged (e.g. a customer/account
                identifier), joined back by index. ``None`` (default)
                returns predictions with no identifier columns.

        Returns:
            pd.DataFrame: Indexed identically to ``df``, with columns
            ``id_columns... , "prediction"`` and, for
            ``task="classification"``, also ``"probability"`` (positive-class
            probability, calibrated if ``calibrator`` is set). Regression
            and clustering outputs never include ``"probability"``.

        Raises:
            ValueError: If ``df`` is empty, is missing a required column, has
                a column whose dtype bucket doesn't match ``schema_``, or if
                an entry in ``id_columns`` isn't present in ``df``.
        """
        self._validate_schema(df)
        if id_columns:
            missing_ids = set(id_columns) - set(df.columns)
            if missing_ids:
                raise ValueError(f"id_columns not present in df: {sorted(missing_ids)}")

        feature_cols = [c for c in df.columns if c in self.schema_]
        transformed = self.feature_pipeline.transform(df[feature_cols])
        X = self.selection_pipeline.transform(transformed)

        scoring_model = (
            self.calibrator.wrap(self.model) if self.calibrator is not None else self.model
        )

        result = pd.DataFrame(index=df.index)
        if id_columns:
            for col in id_columns:
                result[col] = df[col]
        result["prediction"] = scoring_model.predict(X)
        if self.task == "classification":
            result["probability"] = scoring_model.predict_proba(X)[:, 1]
        return result

    def predict_one(self, record: dict) -> dict:
        """Score a single raw record — the natural binding for a future single-request API call.

        Args:
            record (dict): One raw record, e.g. a JSON request body already
                parsed into a dict — keys are raw column names.

        Returns:
            dict: The single scored row from ``predict``, as a dict (keys
            ``"prediction"`` and, for classification, ``"probability"``).

        Raises:
            ValueError: Same conditions as ``predict``.
        """
        result = self.predict(pd.DataFrame([record]))
        return result.iloc[0].to_dict()

    def compute_drift(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Population Stability Index (PSI) between this batch's prediction
        scores and the training-set reference distribution.

        Args:
            df (pd.DataFrame): Raw, unseen data — same requirements as
                ``predict``.

        Returns:
            pd.DataFrame: Columns ``feature`` (str), ``psi`` (float, rounded
            to 4 d.p.), and ``flag`` (bool, ``True`` when ``psi`` exceeds
            ``settings.psi_alert_threshold``). Exactly one row,
            ``feature="__score__"``, representing overall prediction-score
            drift (feature-level PSI is not computed in this version — see
            the design spec's Open Items).

        Raises:
            ValueError: If ``df`` fails schema validation (via ``predict``),
                or if no ``psi_reference_`` was captured at training time
                (always the case for ``task="clustering"``).
        """
        if self.psi_reference_ is None:
            raise ValueError(
                "No psi_reference_ available for this ScoringPipeline — drift "
                "computation requires a training-time reference distribution, which "
                "isn't captured for task='clustering' or if it couldn't be computed "
                "during training."
            )
        self._validate_schema(df)
        feature_cols = [c for c in df.columns if c in self.schema_]
        transformed = self.feature_pipeline.transform(df[feature_cols])
        X = self.selection_pipeline.transform(transformed)

        scoring_model = (
            self.calibrator.wrap(self.model) if self.calibrator is not None else self.model
        )
        if self.task == "classification":
            new_scores = scoring_model.predict_proba(X)[:, 1]
        else:
            new_scores = scoring_model.predict(X)

        psi = psi_score(
            self.psi_reference_.to_numpy(), new_scores, n_bins=settings.psi_feature_n_bins
        )
        return pd.DataFrame(
            [
                {
                    "feature": "__score__",
                    "psi": round(psi, 4),
                    "flag": psi > settings.psi_alert_threshold,
                }
            ]
        )

    def compute_feature_drift(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Characteristic Stability Index (CSI) per raw feature column against
        the training-set reference distribution.

        Unlike ``predict``/``compute_drift``, this method does **not** run the
        fitted ``feature_pipeline``/``selection_pipeline`` transform chain —
        CSI compares raw, business-interpretable feature distributions
        against the training-time raw reference, not encoded/scaled model
        inputs.

        Args:
            df (pd.DataFrame): Raw, unseen data — same requirements as
                ``predict``.

        Returns:
            pd.DataFrame: Columns ``feature`` (str), ``psi`` (float, rounded
            to 4 d.p.), and ``flag`` (bool, ``True`` when exceeding
            ``settings.csi_alert_threshold``) — one row per raw feature
            column shared between ``df`` and ``self.feature_reference_``.
            Contrast with ``compute_drift``'s single ``__score__`` row.

        Raises:
            ValueError: If ``df`` fails schema validation (via ``predict``),
                or if no ``feature_reference_`` was captured at training time
                (always the case for ``task="clustering"``).
        """
        if self.feature_reference_ is None:
            raise ValueError(
                "No feature_reference_ available for this ScoringPipeline — feature drift "
                "computation requires a training-time reference distribution, which isn't "
                "captured for task='clustering' or if it couldn't be computed during "
                "training."
            )
        self._validate_schema(df)
        feature_cols = [c for c in df.columns if c in self.schema_]
        return feature_csi_table(self.feature_reference_, df[feature_cols])

    # ── Validation ───────────────────────────────────────────────────────────

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Validate ``df`` against ``schema_`` before transforming it.

        Args:
            df (pd.DataFrame): Raw input to validate.

        Raises:
            ValueError: If ``df`` is empty, missing a required column, or a
                present column's dtype bucket doesn't match ``schema_``.
        """
        if len(df) == 0:
            raise ValueError("Cannot score an empty DataFrame (0 rows).")

        missing = [col for col in self.schema_ if col not in df.columns]
        if missing:
            raise ValueError(f"df is missing required column(s): {sorted(missing)}")

        mismatched = []
        for col, expected_bucket in self.schema_.items():
            actual_bucket = _dtype_bucket(df[col])
            if actual_bucket != expected_bucket:
                mismatched.append(f"{col!r} (expected {expected_bucket!r}, got {actual_bucket!r})")
        if mismatched:
            raise ValueError(f"df has column(s) with an incompatible dtype: {mismatched}")
