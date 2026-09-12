"""Pydantic request/response models for dscompanion/api/.

Every mutating endpoint follows confirm-before-apply: a `/preview` endpoint returns a
preview object (never persisted), a separate `/confirm` endpoint applies the change.

Schemas exist here for the Steps 1-3 vertical slice actually built (proof
that the architecture works end-to-end). Steps 4-13 get their schemas when
they're ported; their routers are scaffolded but return 501 today.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from dscompanion.api.steps import StepId
from dscompanion.features import ColumnRecipe

__all__ = [
    "RunCreateResponse",
    "RunStateResponse",
    "BackRequest",
    "BackResponse",
    "JobStatusResponse",
    "DataFilesResponse",
    "LoadDataPreviewRequest",
    "LoadDataPreviewResponse",
    "LoadDataConfirmRequest",
    "LoadDataConfirmResponse",
    "TaskTargetPreviewRequest",
    "TaskTargetPreviewResponse",
    "TaskTargetConfirmRequest",
    "TaskTargetConfirmResponse",
    "SplitPreviewRequest",
    "SplitPreviewResponse",
    "SplitConfirmRequest",
    "SplitConfirmResponse",
    "EdaThresholds",
    "EdaPreviewResponse",
    "EdaColumnChartRequest",
    "EdaColumnChartResponse",
    "EdaConfirmRequest",
    "EdaConfirmResponse",
    "TransformerRegistryResponse",
    "FeatureProcessingCandidate",
    "FeatureProcessingPreviewResponse",
    "FeatureProcessingTransformPreviewRequest",
    "FeatureProcessingTransformPreviewResponse",
    "FeatureProcessingConfirmRequest",
    "FeatureProcessingConfirmResponse",
    "FeatureSelectionAuditRow",
    "FeatureSelectionPreviewResponse",
    "FeatureSelectionConfirmRequest",
    "FeatureSelectionConfirmResponse",
    "ImbalancePreviewResponse",
    "ImbalanceConfirmRequest",
    "ImbalanceConfirmResponse",
    "TrainAlgorithmOption",
    "TrainPreviewResponse",
    "TrainConfirmRequest",
    "TrainMetricRow",
    "TrainConfirmResponse",
    "LeaderboardStartResponse",
    "TuningPreviewResponse",
    "TuningStartRequest",
    "TuningStartResponse",
    "TuningConfirmRequest",
    "TuningConfirmResponse",
    "EvaluateMetricRow",
    "EvaluateFlag",
    "EvaluateBaseline",
    "EvaluatePreviewResponse",
    "EvaluateConfirmResponse",
    "CalibrationPreviewResponse",
    "CalibrationConfirmRequest",
    "CalibrationConfirmResponse",
    "ShapPreviewResponse",
    "ShapFeatureRow",
    "ShapRunResponse",
    "ShapConfirmRequest",
    "ShapConfirmResponse",
]


# ── Run-level ────────────────────────────────────────────────────────────────


class RunCreateResponse(BaseModel):
    """Response for ``POST /api/runs``.

    Args:
        run_id (str): The newly created run's identifier — pass this in every
            subsequent request's path.
    """

    run_id: str


class RunStateResponse(BaseModel):
    """Response for ``GET /api/runs/{run_id}/state`` — the REST equivalent of
    ``_interactive_flags()`` + ``_build_partial_config_dict()`` combined.

    Args:
        run_id (str): This run's identifier.
        step_status (dict[StepId, str]): One of ``"done"``, ``"current"``,
            ``"pending"``, ``"flagged"`` per step.
        step_confirmed (dict[StepId, bool]): Whether each step has been
            confirmed.
        audit_trail (list[tuple[StepId, str]]): One plain-language entry per
            confirmed step, in step order.
        config_preview (dict[str, Any]): Flat dict keyed by step slug, containing
            only the fields decided by confirmed steps so far — grows exactly in
            step with confirmations, same philosophy as ``streamlit_app.py``'s
            ``_build_partial_config_dict()``.
    """

    run_id: str
    step_status: dict[StepId, str]
    step_confirmed: dict[StepId, bool]
    audit_trail: list[tuple[StepId, str]]
    config_preview: dict[str, Any]


class BackRequest(BaseModel):
    """Request for ``POST /api/runs/{run_id}/back``.

    Args:
        to_step (StepId): The step to go back to — this step and every step
            after it will be invalidated once confirmed.
        confirmed (bool): ``False`` (default) for the first call — the response
            will ask for confirmation without changing anything.  ``True`` to
            actually apply the reset, mirroring the UI's two-phase back button.
    """

    to_step: StepId
    confirmed: bool = False


class BackResponse(BaseModel):
    """Response for ``POST /api/runs/{run_id}/back``.

    Args:
        requires_confirmation (bool): ``True`` when this call was the first
            phase (``confirmed=False`` in the request) — call again with
            ``confirmed=True`` to actually apply the reset.
        state (RunStateResponse | None): The updated state, populated only once
            the reset has actually been applied (``requires_confirmation=False``).
    """

    requires_confirmation: bool
    state: RunStateResponse | None = None


class JobStatusResponse(BaseModel):
    """Response for ``GET /api/runs/{run_id}/jobs/{job_id}`` — polled by the
    frontend the same way Step 8's ``st.fragment(run_every=1)`` polls
    ``int.step8.lb_progress`` today.

    Args:
        job_id (str): This job's identifier.
        step (StepId): Which wizard step started this job (``"train"`` or
            ``"tuning"``).
        status (str): One of ``"running"``, ``"done"``, ``"error"``.
        progress (list[dict]): Incremental progress entries — same
            ``{"algorithm"/"trial", "status", "detail"}`` shape Step 8's
            worker already produces.
        result (dict | None): Final result payload once ``status != "running"``.
        error (str | None): Error message if ``status == "error"``.
    """

    job_id: str
    step: StepId
    status: Literal["running", "done", "error"]
    progress: list[dict]
    result: dict | None = None
    error: str | None = None


class DataFilesResponse(BaseModel):
    """Response for ``GET /api/data-files`` — lists server-side files Step 1 can load.

    Not run-scoped (no ``run_id`` in the path) since it doesn't touch any run's state —
    mirrors ``dscompanion/app/interactive_step1.py``'s ``_list_data_files()``, which the browser
    can't do itself (no filesystem access). Browser file upload is a separate, deferred
    concern.

    Args:
        files (list[str]): Relative paths (from the data root) of every ``.csv``/
            ``.parquet``/``.xlsx`` file found, sorted.
    """

    files: list[str]


# ── Step 1 — Load Data ───────────────────────────────────────────────────────


class LoadDataPreviewRequest(BaseModel):
    """Request for ``POST /api/runs/{run_id}/steps/1/preview``.

    Args:
        path (str): Path to the dataset file (server-side path or an uploaded
            file's staged location — upload transport is a separate concern).
        format (str): One of ``"csv"``, ``"parquet"``, ``"xlsx"``.
        sheet_name (str | None): XLSX sheet name. Ignored for other formats.
    """

    path: str
    format: Literal["csv", "parquet", "xlsx"]
    sheet_name: str | None = None


class LoadDataPreviewResponse(BaseModel):
    """Response for ``POST /api/runs/{run_id}/steps/1/preview``.

    Args:
        columns (list[str]): Column names as loaded.
        dtypes (dict[str, str]): Column name to pandas dtype string.
        n_rows (int): Row count.
        duplicate_columns (list[str]): Column names that appear more than once
            in the raw file header — must be resolved before confirming.
        sample_rows (list[dict[str, Any]]): The first ``api_settings.preview_sample_rows``
            rows, JSON-safe (NaN/NaT become ``None``, Timestamps become ISO strings) —
            mirrors ``interactive_step1.py``'s ``st.dataframe(df.head(10))``.
    """

    columns: list[str]
    dtypes: dict[str, str]
    n_rows: int
    duplicate_columns: list[str]
    sample_rows: list[dict[str, Any]]


class LoadDataConfirmRequest(LoadDataPreviewRequest):
    """Request for ``POST /api/runs/{run_id}/steps/1/confirm``.

    Args:
        duplicate_resolutions (dict[str, str]): Per-duplicate-occurrence choice
            — ``{"original_col_name__2": "rename:new_name"}`` or
            ``{"original_col_name__2": "drop"}``. Empty when
            ``duplicate_columns`` was empty at preview time.
    """

    duplicate_resolutions: dict[str, str] = Field(default_factory=dict)


class LoadDataConfirmResponse(BaseModel):
    """Response for ``POST /api/runs/{run_id}/steps/1/confirm``.

    Args:
        confirmed (bool): Always ``True`` on success.
        columns (list[str]): Final column names after duplicate resolution.
        n_rows (int): Row count.
    """

    confirmed: bool
    columns: list[str]
    n_rows: int


# ── Step 2 — Task & Target ───────────────────────────────────────────────────


class TaskTargetPreviewRequest(BaseModel):
    """Request for ``POST /api/runs/{run_id}/steps/2/preview``.

    Args:
        task (str): Currently only ``"classification"`` is supported (mirrors
            ``interactive_step2.py``'s current restriction).
        target (str): Target column name — must be present in Step 1's columns.
        identifier_columns (list[str]): Columns to exclude as features (IDs,
            not real signal).
    """

    task: Literal["classification"]
    target: str
    identifier_columns: list[str] = Field(default_factory=list)


class TaskTargetPreviewResponse(BaseModel):
    """Response for ``POST /api/runs/{run_id}/steps/2/preview``.

    Args:
        valid (bool): Whether the target passes dscompanion's validation (present,
            no missing values, strictly binary {0, 1}, no degenerate minority
            class).
        errors (list[str]): Validation failure messages. Empty when
            ``valid=True``.
        target_value_counts (dict[str, int]): Class counts for the chosen
            target, for display.
    """

    valid: bool
    errors: list[str]
    target_value_counts: dict[str, int]


class TaskTargetConfirmRequest(TaskTargetPreviewRequest):
    """Request for ``POST /api/runs/{run_id}/steps/2/confirm`` — identical
    shape to the preview request; confirming re-validates before persisting.
    """


class TaskTargetConfirmResponse(BaseModel):
    """Response for ``POST /api/runs/{run_id}/steps/2/confirm``.

    Args:
        confirmed (bool): Always ``True`` on success.
        task (str): The confirmed task.
        target (str): The confirmed target column.
    """

    confirmed: bool
    task: str
    target: str


# ── Step 3 — Split ───────────────────────────────────────────────────────────


class SplitPreviewRequest(BaseModel):
    """Request for ``POST /api/runs/{run_id}/steps/3/preview``.

    Args:
        method (str): One of ``"stratified"``, ``"random"``, ``"temporal"``,
            ``"grouped"``.
        test_size (float): Fraction reserved for the test set.
        val_size (float): Fraction of the training portion reserved for
            validation.
        date_col (str | None): Required when ``method="temporal"``.
        group_col (str | None): Required when ``method="grouped"``.
        oot_cutoff (str | None): ISO date string — required when
            ``method="temporal"``.
    """

    method: Literal["stratified", "random", "temporal", "grouped"]
    test_size: float = 0.2
    val_size: float = 0.1
    date_col: str | None = None
    group_col: str | None = None
    oot_cutoff: str | None = None


class SplitPreviewResponse(BaseModel):
    """Response for ``POST /api/runs/{run_id}/steps/3/preview``.

    Args:
        row_counts (dict[str, int]): Row count per split partition (``train``,
            ``val``, ``test``, ``oot``).
        event_rates (dict[str, float]): Target event rate per partition, for
            classification tasks — lets the user sanity-check the split before
            applying it.
    """

    row_counts: dict[str, int]
    event_rates: dict[str, float]


class SplitConfirmRequest(SplitPreviewRequest):
    """Request for ``POST /api/runs/{run_id}/steps/3/confirm`` — identical
    shape to the preview request; confirming re-runs the split before
    persisting (the split itself is cheap, unlike Steps 8/9).
    """


class SplitConfirmResponse(BaseModel):
    """Response for ``POST /api/runs/{run_id}/steps/3/confirm``.

    Args:
        confirmed (bool): Always ``True`` on success.
        row_counts (dict[str, int]): Row count per split partition.
    """

    confirmed: bool
    row_counts: dict[str, int]


# ── Step 4 — Explore the Data (EDA) ─────────────────────────────────────────


class EdaThresholds(BaseModel):
    """The actual threshold values used to compute this preview — returned so the
    frontend's explanatory captions never hardcode (and drift from) ``dscompanion.config.settings``.

    Args:
        near_zero_variance (float): ``settings.near_zero_variance_threshold``.
        high_cardinality (int): ``settings.high_cardinality_threshold``.
        correlation (float): ``settings.correlation_threshold``.
        vif (float): ``settings.vif_threshold``.
    """

    near_zero_variance: float
    high_cardinality: int
    correlation: float
    vif: float


class EdaPreviewResponse(BaseModel):
    """Response for ``POST /api/runs/{run_id}/steps/4/preview`` — read-only, computes
    (and caches server-side) the full ``EDAReport`` in one call. Per-column charts
    (numeric KDE, categorical bar, bivariate target-rate) are fetched separately via
    ``POST .../steps/4/column-chart`` rather than included here, since only one column
    is ever viewed at a time — mirrors ``interactive_step4.py``'s selectbox-driven charts.

    Args:
        overview (dict[str, Any]): ``EDAReport.overview_summary()`` — row/column/missing/
            duplicate counts.
        missing_values_chart (dict[str, Any]): Plotly figure (JSON-safe via
            ``fig.to_json()``) — missing % by feature, sorted descending.
        missing_correlation_heatmap (dict[str, Any] | None): Plotly figure, or ``None``
            when there's nothing to show (fewer than 2 partially-missing columns).
        high_missing_alerts (list[dict[str, str]]): ``{"feature", "detail"}`` entries
            from ``EDAReport.alerts()`` filtered to ``HIGH_MISSING`` type.
        numeric_summary (list[dict[str, Any]]): ``EDAReport.numeric_summary()`` as records.
        categorical_summary (list[dict[str, Any]]): ``EDAReport.categorical_summary()``
            as records.
        iv_table (list[dict[str, Any]]): ``EDAReport.iv_table()`` as records, IV-ranked.
        pearson_heatmap (dict[str, Any] | None): Plotly figure, or ``None`` when fewer
            than 2 numeric columns exist.
        high_correlation_pairs (list[dict[str, Any]]): ``EDAReport.flag_high_correlation()``
            as records. Empty when no pairs exceed the threshold.
        vif_table (list[dict[str, Any]]): ``EDAReport.vif_table()`` as records. Empty
            when fewer than 2 numeric columns exist.
        cramers_v_table (list[dict[str, Any]]): ``EDAReport.cramers_v_table()`` as
            records. Empty when fewer than 2 categorical columns exist.
        thresholds (EdaThresholds): The actual threshold values used, for captions.
    """

    overview: dict[str, Any]
    missing_values_chart: dict[str, Any]
    missing_correlation_heatmap: dict[str, Any] | None
    high_missing_alerts: list[dict[str, str]]
    numeric_summary: list[dict[str, Any]]
    categorical_summary: list[dict[str, Any]]
    iv_table: list[dict[str, Any]]
    pearson_heatmap: dict[str, Any] | None
    high_correlation_pairs: list[dict[str, Any]]
    vif_table: list[dict[str, Any]]
    cramers_v_table: list[dict[str, Any]]
    thresholds: EdaThresholds


class EdaColumnChartRequest(BaseModel):
    """Request for ``POST /api/runs/{run_id}/steps/4/column-chart``.

    Args:
        kind (str): Which per-column chart to build — mirrors
            ``interactive_step4.py``'s three selectbox-driven charts.
        column (str): The column (or feature, for ``bivariate_rate``) to chart.
    """

    kind: Literal["numeric_kde", "categorical_bar", "bivariate_rate"]
    column: str


class EdaColumnChartResponse(BaseModel):
    """Response for ``POST /api/runs/{run_id}/steps/4/column-chart``.

    Args:
        figure (dict[str, Any]): Plotly figure, JSON-safe via ``fig.to_json()``.
        caption (str | None): Extra context only ``bivariate_rate`` sets — the IV
            predictive-power gloss (e.g. ``"predictive power: Strong"``).
    """

    figure: dict[str, Any]
    caption: str | None = None


class EdaConfirmRequest(BaseModel):
    """Request for ``POST /api/runs/{run_id}/steps/4/confirm``.

    Args:
        skipped (bool): ``True`` when the user turned off "Run a data quality check"
            entirely — mirrors ``interactive_step4.py``'s ``run_eda`` toggle. When
            ``True``, no ``EDAReport`` is required to have been previewed first.
    """

    skipped: bool = False


class EdaConfirmResponse(BaseModel):
    """Response for ``POST /api/runs/{run_id}/steps/4/confirm``.

    Args:
        confirmed (bool): Always ``True`` on success.
        status (str): ``"flagged"`` when skipped or the check failed, ``"done"``
            otherwise — mirrors ``interactive_step4.py``'s ``set_step_status`` calls.
    """

    confirmed: bool
    status: Literal["done", "flagged"]


# ── Transformer registry (not run-scoped) ────────────────────────────────────


class TransformerRegistryResponse(BaseModel):
    """Response for ``GET /api/transformer-registry`` — the menu Step 5's recipe editor
    builds its "add step" control from.

    Args:
        names (list[str]): Every registered transformer key (e.g. ``"log"``,
            ``"clip_lower"``, ``"onehot_encode"``) — valid values for
            ``TransformStep.transformer``.
    """

    names: list[str]


# ── Step 5 — Feature Processing (ColumnRecipe decisions) ────────────────────


class FeatureProcessingCandidate(BaseModel):
    """One feature column's recommendation, for Step 5's preview.

    Args:
        column (str): Column name.
        is_numeric (bool): Which summary table (and recommender rules) produced
            ``recommended`` — also drives the frontend's "Reject" safety-warning text.
        missing_rows (int): Missing-row count for this column on the training set.
        missing_pct (float): Missing-row percentage (0-100 scale).
        suggested_reasons (list[str]): Non-empty when Step 4's EDA flagged this column
            for a reason other than missingness (e.g. ``"constant"``, ``"low_iv"``) —
            empty when Step 4 was skipped/failed or nothing was flagged.
        needs_attention (bool): ``True`` if this column belongs in the "Needing
            Attention" group (has missing values or an EDA red flag); ``False`` for
            the "Rest" group. Lets the frontend render both groups from one flat list.
        recommended (ColumnRecipe): The theory-driven recommendation
            (``recommend_column_recipe()``) — ``source="recommended"``, possibly zero
            steps for a clean column.
    """

    column: str
    is_numeric: bool
    missing_rows: int
    missing_pct: float
    suggested_reasons: list[str]
    needs_attention: bool
    recommended: ColumnRecipe


class FeatureProcessingPreviewResponse(BaseModel):
    """Response for ``POST /api/runs/{run_id}/steps/feature_processing/preview``.

    Args:
        candidates (list[FeatureProcessingCandidate]): One entry per feature column
            eligible for a recipe decision — excludes the date column (Step 3),
            identifier columns (Step 2), and any boolean/datetime-dtype column (out
            of scope for recipe treatment, matching ``DateFeatureExtractor`` being a
            separate mechanism). Ordered "Needing Attention" columns first, then
            "Rest", matching ``interactive_step_feature_processing.py``'s grouping.
    """

    candidates: list[FeatureProcessingCandidate]


class FeatureProcessingTransformPreviewRequest(BaseModel):
    """Request for ``POST /api/runs/{run_id}/steps/feature_processing/transform-preview``.

    Args:
        recipes (dict[str, ColumnRecipe]): The client's *current* (possibly edited)
            accepted recipes — only columns the user has set to "Accept" belong here,
            matching ``_render_live_preview()``'s ``accepted`` dict.
    """

    recipes: dict[str, ColumnRecipe]


class FeatureProcessingTransformPreviewResponse(BaseModel):
    """Response for ``POST /api/runs/{run_id}/steps/feature_processing/transform-preview``.

    Args:
        before (list[dict[str, Any]]): First 5 training rows, accepted columns only,
            JSON-safe records.
        after (list[dict[str, Any]]): Same 5 rows after ``FeatureTransformChain``
            fit_transform — column-expanding steps (e.g. one-hot) show their expanded
            output columns, so ``after``'s keys may not match ``before``'s.
        columns_before (list[str]): ``before``'s column order.
        columns_after (list[str]): ``after``'s column order.
    """

    before: list[dict[str, Any]]
    after: list[dict[str, Any]]
    columns_before: list[str]
    columns_after: list[str]


class FeatureProcessingConfirmRequest(BaseModel):
    """Request for ``POST /api/runs/{run_id}/steps/feature_processing/confirm``.

    Args:
        recipes (dict[str, ColumnRecipe]): Final accepted recipes (accepted columns
            only) — becomes this run's canonical ``feature_processing_recipes``.
        dropped_columns (list[str]): Columns explicitly chosen "Drop" — removed
            entirely, never reaching ``FeatureTransformChain``.
    """

    recipes: dict[str, ColumnRecipe]
    dropped_columns: list[str] = Field(default_factory=list)


class FeatureProcessingConfirmResponse(BaseModel):
    """Response for ``POST /api/runs/{run_id}/steps/feature_processing/confirm``.

    Args:
        confirmed (bool): Always ``True`` on success.
        n_accepted (int): Number of columns with a non-empty accepted recipe.
        n_dropped (int): Number of columns dropped entirely.
        n_rejected (int): Number of columns left as raw passthrough (present in the
            candidate set but absent from both ``recipes`` and ``dropped_columns``).
    """

    confirmed: bool
    n_accepted: int
    n_dropped: int
    n_rejected: int


# ── Step 6 — Feature Selection ───────────────────────────────────────────────


class FeatureSelectionAuditRow(BaseModel):
    """One proposed-removal row from ``FeatureSelectionPipeline.audit_report()``.

    Args:
        feature (str): Feature (post-Step-5-transform) column name.
        removed_by (str): Selector class name that flagged this feature (e.g.
            ``"CorrelationSelector"``) — the frontend maps this to a plain-language
            stage label, mirroring ``interactive_step_feature_selection.py``'s
            ``_STAGE_LABELS``.
        reason (str): Plain-language reason for the flag.
        original_position (int): 0-based column index in the transformed input, or
            ``-1`` if not found.
    """

    feature: str
    removed_by: str
    reason: str
    original_position: int


class FeatureSelectionPreviewResponse(BaseModel):
    """Response for ``POST /api/runs/{run_id}/steps/feature_selection/preview``.

    Args:
        audit (list[FeatureSelectionAuditRow]): One row per feature flagged for
            removal — empty when nothing was flagged.
        waterfall_chart (dict[str, Any] | None): Plotly figure (JSON-safe via
            ``fig.to_json()``), or ``None`` when ``audit`` is empty (nothing to plot).
        total_features (int): Column count of the Step-5-transformed feature set —
            used to detect the "every feature flagged" edge case client-side.
    """

    audit: list[FeatureSelectionAuditRow]
    waterfall_chart: dict[str, Any] | None
    total_features: int


class FeatureSelectionConfirmRequest(BaseModel):
    """Request for ``POST /api/runs/{run_id}/steps/feature_selection/confirm``.

    Args:
        removed_columns (list[str]): Features to actually remove — a subset of (or
            equal to) the preview's flagged set, since the user can uncheck any row.
        skipped (bool): ``True`` when selection itself failed and the user chose to
            continue without it — mirrors ``interactive_step_feature_selection.py``'s
            "Continue without feature selection" recovery path. When ``True``,
            ``removed_columns`` is ignored (treated as empty).
    """

    removed_columns: list[str] = Field(default_factory=list)
    skipped: bool = False


class FeatureSelectionConfirmResponse(BaseModel):
    """Response for ``POST /api/runs/{run_id}/steps/feature_selection/confirm``.

    Args:
        confirmed (bool): Always ``True`` on success.
        status (str): ``"flagged"`` when skipped, ``"done"`` otherwise.
        removed_columns (list[str]): The columns actually persisted as removed.
    """

    confirmed: bool
    status: Literal["done", "flagged"]
    removed_columns: list[str]


# ── Step 7 — Class Imbalance Handling ────────────────────────────────────────


class ImbalancePreviewResponse(BaseModel):
    """Response for ``POST /api/runs/{run_id}/steps/imbalance/preview``.

    Args:
        applicable (bool): ``False`` for non-classification tasks — imbalance
            handling only applies to classification, matching
            ``interactive_step_imbalance.py``'s early-exit.
        n_train (int): Training-set row count.
        event_rate (float): Fraction of training rows with target ``1``.
        is_balanced (bool): Whether ``event_rate`` falls within
            ``settings.imbalance_balanced_min_event_rate``'s symmetric band.
        class_counts (dict[str, int]): ``{"0": n_zeros, "1": n_ones}``.
        balanced_min_event_rate (float): The actual threshold used, for captions —
            never hardcoded in the frontend.
    """

    applicable: bool
    n_train: int
    event_rate: float
    is_balanced: bool
    class_counts: dict[str, int]
    balanced_min_event_rate: float


class ImbalanceConfirmRequest(BaseModel):
    """Request for ``POST /api/runs/{run_id}/steps/imbalance/confirm``.

    Args:
        strategy (str): One of ``"class_weight"``, ``"smote"``, ``"undersample"``,
            ``"none"``. ``"oversample"`` is deliberately excluded — accepted by
            ``ImbalanceConfig`` but unhandled by ``ImbalanceHandler._build_sampler()``,
            so offering it would let the user choose a path that breaks at training
            time (Step 8).
    """

    strategy: Literal["class_weight", "smote", "undersample", "none"]


class ImbalanceConfirmResponse(BaseModel):
    """Response for ``POST /api/runs/{run_id}/steps/imbalance/confirm``.

    Args:
        confirmed (bool): Always ``True`` on success.
        strategy (str): The confirmed strategy.
        rows_before (int): Training-set row count before resampling.
        rows_after (int | None): Estimated post-resample row count for
            ``"smote"``/``"undersample"`` (``sampling_strategy="auto"`` assumption,
            balancing both classes to the same count); ``None`` for
            ``"class_weight"``/``"none"``, which never change row counts.
    """

    confirmed: bool
    strategy: str
    rows_before: int
    rows_after: int | None


# ── Step 8 — Train a Model ───────────────────────────────────────────────────


class TrainAlgorithmOption(BaseModel):
    """One selectable algorithm for Step 8's picker.

    Args:
        name (str): Algorithm key (e.g. ``"xgboost"``), as accepted by
            ``ModelFactory.build()``.
        note (str): Plain-language one-liner shown next to it.
    """

    name: str
    note: str


class TrainPreviewResponse(BaseModel):
    """Response for ``POST /api/runs/{run_id}/steps/train/preview``.

    Args:
        algorithms (list[TrainAlgorithmOption]): Every algorithm available for
            the single-algorithm path (task-appropriate).
        leaderboard_algorithms (list[str]): Subset of ``algorithms`` used for
            the leaderboard comparison — excludes ``"svm"`` (too slow for an
            automated multi-algorithm run).
        n_features (int): Feature-column count after Steps 2/3/5/6 exclusions,
            before any transform runs.
    """

    algorithms: list[TrainAlgorithmOption]
    leaderboard_algorithms: list[str]
    n_features: int


class TrainConfirmRequest(BaseModel):
    """Request for ``POST /api/runs/{run_id}/steps/train/confirm``.

    Args:
        algorithm (str): The algorithm to train and confirm (single-algorithm
            path), or the leaderboard winner's name (leaderboard path — the
            already-trained winner is read from ``run.artifacts`` rather than
            retrained).
    """

    algorithm: str


class TrainMetricRow(BaseModel):
    """One row of Step 8's brief AUC/Gini/KS summary table.

    Args:
        split (str): One of ``"train"``, ``"val"``, ``"test"``, ``"oot"``.
        auc_roc (float | None): AUC-ROC, ``None`` if unavailable for this split.
        gini (float | None): Gini coefficient, ``None`` if unavailable.
        ks_statistic (float | None): KS statistic, ``None`` if unavailable.
    """

    split: str
    auc_roc: float | None = None
    gini: float | None = None
    ks_statistic: float | None = None


class TrainConfirmResponse(BaseModel):
    """Response for ``POST /api/runs/{run_id}/steps/train/confirm``.

    Args:
        confirmed (bool): Always ``True`` on success.
        algorithm (str): The confirmed algorithm.
        metrics (list[TrainMetricRow]): Brief per-split metric summary.
    """

    confirmed: bool
    algorithm: str
    metrics: list[TrainMetricRow]


class LeaderboardStartResponse(BaseModel):
    """Response for ``POST /api/runs/{run_id}/steps/train/leaderboard/start``.

    Args:
        job_id (str): Poll ``GET /api/runs/{run_id}/jobs/{job_id}`` with this
            id for live per-algorithm progress and the final ranked table
            (``result`` shape: ``{"rows": [{"algorithm", "status", "fit_time_s",
            "roc_auc", "error"}, ...], "winner": str | None}``).
    """

    job_id: str


# ── Step 9 — Hyperparameter Tuning ───────────────────────────────────────────


class TuningPreviewResponse(BaseModel):
    """Response for ``POST /api/runs/{run_id}/steps/tuning/preview``.

    Args:
        algorithm (str): The algorithm confirmed at Step 8.
        available (bool): ``False`` when ``algorithm`` has no Optuna search
            space (``gradient_boosting``/``lightgbm``) — the frontend should
            auto-skip straight to confirming with ``use_tuned=False``,
            matching Streamlit's silent auto-skip rather than offering a
            "Start Tuning" control that would only fail.
        default_n_trials (int): Suggested starting value for the trials slider.
        min_n_trials (int): Slider minimum.
        max_n_trials (int): Slider maximum.
    """

    algorithm: str
    available: bool
    default_n_trials: int
    min_n_trials: int
    max_n_trials: int


class TuningStartRequest(BaseModel):
    """Request for ``POST /api/runs/{run_id}/steps/tuning/tune/start``.

    Args:
        n_trials (int): Number of Optuna trials to run.
    """

    n_trials: int


class TuningStartResponse(BaseModel):
    """Response for ``POST /api/runs/{run_id}/steps/tuning/tune/start``.

    Args:
        job_id (str): Poll ``GET /api/runs/{run_id}/jobs/{job_id}`` with this
            id — ``result`` holds the full comparison payload once done:
            ``{"best_score": float | None, "baseline_auc": float | None,
            "params_comparison": [{"parameter", "default", "tuned", "changed"},
            ...], "trials": [{"trial_number", "metric_value"}, ...],
            "optimization_curve": <Plotly figure JSON | None>}``. ``baseline_auc``
            is Step 8's untuned model re-evaluated fresh (val split, falling back
            to test), for the before/after comparison.
    """

    job_id: str


class TuningConfirmRequest(BaseModel):
    """Request for ``POST /api/runs/{run_id}/steps/tuning/confirm``.

    Args:
        use_tuned (bool): ``True`` to carry the tuned model forward, ``False``
            to keep Step 8's model unchanged — covers the skip-toggle,
            auto-skip, and reject-tuned-parameters paths alike, all three
            confirm with ``use_tuned=False``.
    """

    use_tuned: bool


class TuningConfirmResponse(BaseModel):
    """Response for ``POST /api/runs/{run_id}/steps/tuning/confirm``.

    Args:
        confirmed (bool): Always ``True`` on success.
        tuned (bool): Whether the tuned model was accepted.
        best_score (float | None): Best AUC found by tuning, if applicable.
    """

    confirmed: bool
    tuned: bool
    best_score: float | None = None


# ── Step 10 — Evaluate ────────────────────────────────────────────────────────


class EvaluateMetricRow(BaseModel):
    """One ``(split, metric)`` pair from Step 10's full metric breakdown.

    Args:
        split (str): One of ``"train"``, ``"val"``, ``"test"``, ``"oot"``.
        metric (str): Metric key (e.g. ``"roc_auc"``).
        label (str): Plain-language display label (e.g. ``"AUC-ROC"``).
        gloss (str): Plain-language explanation of what the metric means.
        value (float | None): Metric value, ``None`` if unavailable or ``NaN``
            for this split.
    """

    split: str
    metric: str
    label: str
    gloss: str
    value: float | None = None


class EvaluateFlag(BaseModel):
    """One suspicious-pattern warning from Step 10's evaluation.

    Args:
        level (str): One of ``"error"``, ``"warning"``, ``"info"``.
        message (str): Plain-language Markdown warning text.
    """

    level: str
    message: str


class EvaluateBaseline(BaseModel):
    """Step 8's untuned baseline value for one headline metric, for the
    tuned-vs-default delta shown when Step 9's tuning was accepted.

    Args:
        metric (str): Metric key (e.g. ``"roc_auc"``).
        value (float | None): Step 8's untuned model's value on
            ``primary_split``, ``None`` if unavailable.
    """

    metric: str
    value: float | None = None


class EvaluatePreviewResponse(BaseModel):
    """Response for ``POST /api/runs/{run_id}/steps/evaluate/preview``.

    Args:
        algorithm (str): The confirmed algorithm from Step 8.
        tuned (bool): Whether Step 9's tuned model was accepted — controls
            whether ``baseline`` is populated.
        primary_split (str): The split used for headline metrics and flags —
            ``"val"`` if available, else ``"test"``, else ``"train"``.
        metrics (list[EvaluateMetricRow]): Full per-split metric breakdown,
            headline metrics (``roc_auc``, ``gini``, ``ks_statistic``) first,
            then the remaining supporting metrics.
        baseline (list[EvaluateBaseline]): Step 8's untuned headline-metric
            values on ``primary_split``, for the "vs. default" delta shown
            when tuned. Empty when ``tuned=False``.
        flags (list[EvaluateFlag]): Suspicious-pattern warnings. Empty when
            none were found.
    """

    algorithm: str
    tuned: bool
    primary_split: str
    metrics: list[EvaluateMetricRow] = Field(default_factory=list)
    baseline: list[EvaluateBaseline] = Field(default_factory=list)
    flags: list[EvaluateFlag] = Field(default_factory=list)


class EvaluateConfirmResponse(BaseModel):
    """Response for ``POST /api/runs/{run_id}/steps/evaluate/confirm``.

    Args:
        confirmed (bool): Always ``True`` on success.
    """

    confirmed: bool


# ── Step 11 — Calibration ─────────────────────────────────────────────────────


class CalibrationPreviewResponse(BaseModel):
    """Response for ``POST /api/runs/{run_id}/steps/calibration/preview``.

    Args:
        available (bool): ``False`` for non-classification tasks — the
            frontend should auto-skip straight to confirming with
            ``apply=False``, matching Streamlit's silent auto-skip rather
            than offering a choice that has no effect.
        split_used (str | None): Held-out split the calibrator was fit and
            evaluated on (``"val"`` if available, else ``"test"``, else
            ``"train"``). ``None`` when ``available=False``.
        ece_before (float | None): Expected Calibration Error before
            calibration. ``None`` when ``available=False``.
        ece_after (float | None): Expected Calibration Error after isotonic
            calibration. ``None`` when ``available=False``.
        recommend_apply (bool): ``True`` when calibration meaningfully
            reduced ECE — drives which of the two choice buttons the
            frontend highlights as primary. Always ``False`` when
            ``available=False``.
    """

    available: bool
    split_used: str | None = None
    ece_before: float | None = None
    ece_after: float | None = None
    recommend_apply: bool = False


class CalibrationConfirmRequest(BaseModel):
    """Request for ``POST /api/runs/{run_id}/steps/calibration/confirm``.

    Args:
        apply (bool): ``True`` to wrap the model with calibrated
            probabilities, ``False`` to keep the raw scores unchanged.
            Ignored (treated as ``False``) for non-classification tasks.
    """

    apply: bool


class CalibrationConfirmResponse(BaseModel):
    """Response for ``POST /api/runs/{run_id}/steps/calibration/confirm``.

    Args:
        confirmed (bool): Always ``True`` on success.
        applied (bool): Whether calibration was actually applied — ``False``
            for a non-classification task regardless of the request.
    """

    confirmed: bool
    applied: bool


# ── Step 12 — Explainability (SHAP) ───────────────────────────────────────────


class ShapPreviewResponse(BaseModel):
    """Response for ``POST /api/runs/{run_id}/steps/shap/preview``.

    Args:
        available (bool): Whether SHAP can be run for this run. Always
            ``True`` today — exists as a forward-compatible kill switch.
    """

    available: bool


class ShapFeatureRow(BaseModel):
    """One row of Step 12's top-feature importance table.

    Args:
        rank (int): 1-based rank, most important feature first.
        feature (str): Feature name.
        mean_abs_shap (float): Mean absolute SHAP value across the sample.
    """

    rank: int
    feature: str
    mean_abs_shap: float


class ShapRunResponse(BaseModel):
    """Response for ``POST /api/runs/{run_id}/steps/shap/run``.

    Args:
        figure (dict[str, Any]): SHAP summary bar chart, JSON-safe via
            ``fig.to_json()``.
        top_features (list[ShapFeatureRow]): Top 10 features by mean
            absolute SHAP value.
        split_used (str): Held-out split SHAP was computed on (``"val"`` if
            available, else ``"test"``, else ``"train"``).
        n_rows (int): Number of rows actually sampled for the computation
            (capped at ``settings.shap_interactive_max_rows``).
    """

    figure: dict[str, Any]
    top_features: list[ShapFeatureRow]
    split_used: str
    n_rows: int


class ShapConfirmRequest(BaseModel):
    """Request for ``POST /api/runs/{run_id}/steps/shap/confirm``.

    Args:
        shap_run (bool): Whether SHAP was actually computed (``True``) or
            skipped by the user (``False``) — recorded in the audit trail.
    """

    shap_run: bool


class ShapConfirmResponse(BaseModel):
    """Response for ``POST /api/runs/{run_id}/steps/shap/confirm``.

    Args:
        confirmed (bool): Always ``True`` on success.
    """

    confirmed: bool
