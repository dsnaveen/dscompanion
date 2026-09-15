"""PipelineConfig: pydantic schema for a full dscompanion experiment YAML.

A scientist writes a YAML, passes it to PipelineRunner, and gets a
complete model artefact. Minimum required fields: name, data.path,
data.target, model.task. Everything else has a production-safe default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from dscompanion.features.transform_chain import ColumnRecipe

__all__ = [
    "PipelineConfig",
    "DataConfig",
    "SplitConfig",
    "EDAConfig",
    "ImputerConfig",
    "EncoderConfig",
    "ScalerConfig",
    "DistributionConfig",
    "FeaturesConfig",
    "ImbalanceConfig",
    "TargetConfig",
    "SelectionConfig",
    "ModelConfig",
    "TuningConfig",
    "LeaderboardConfig",
    "ExplainConfig",
    "ReportingConfig",
]

# ── Data ─────────────────────────────────────────────────────────────────────


class DataConfig(BaseModel):
    """Specifies where to read data and which column is the prediction target.

    Supports parquet, CSV, Excel, and Delta Lake paths.  All storage paths
    should use ``abfss://`` on Databricks.  Set ``nrows`` to a small number
    during local development to avoid loading the full dataset.

    Args:
        path (str): Full path to the input data file or Delta table.
            On Databricks use ``abfss://container@account.dfs.core.windows.net/...``.
        format (str): File format.  One of ``"parquet"``, ``"csv"``,
            ``"excel"``, ``"delta"``.  Defaults to ``"parquet"``.
        target (str): Name of the target column in the dataset.
        feature_columns (list[str] | None): Explicit list of feature columns
            to use.  When ``None`` (default), all columns except ``target``
            and ``date_column`` are used as features.
        date_column (str | None): Name of a date/timestamp column used only
            for temporal splitting.  Not included in features.
        nrows (int | None): If set, only the first ``nrows`` rows are loaded.
            Use during local development only — never set this in production.
        sheet_name (str | int | list[str | int] | None): Only used when
            ``format="excel"``; ignored otherwise.  A single sheet name/index
            reads that one sheet.  A list of names/indices reads each and
            vertically concatenates them into one DataFrame — every sheet in
            the list must have identical columns, or loading raises a clear
            error rather than silently producing NaN-filled columns.
            ``None`` (default) reads the first sheet, matching how
            ``format="parquet"``/``"csv"`` each read one dataset from one
            path.

    Returns:
        DataConfig: Validated data specification.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    format: str = "parquet"
    target: str
    feature_columns: list[str] | None = None
    date_column: str | None = None
    nrows: int | None = None
    sheet_name: str | int | list[str | int] | None = None

    @field_validator("format")
    @classmethod
    def valid_format(cls, v: str) -> str:
        allowed = {"parquet", "csv", "excel", "delta"}
        if v not in allowed:
            raise ValueError(f"format must be one of {allowed}, got {v!r}")
        return v


# ── Split ─────────────────────────────────────────────────────────────────────


class SplitConfig(BaseModel):
    """Controls how the dataset is split into train / validation / test sets.

    The ``method`` field determines the splitting strategy.  ``temporal``
    requires ``data.date_column`` to be set.  ``grouped`` requires
    ``group_column`` to be set here.

    Args:
        method (str): Splitting strategy.  One of ``"stratified"`` (default),
            ``"random"``, ``"temporal"``, ``"grouped"``.
        test_size (float): Fraction of total data reserved for the test set.
            Must be in ``(0, 1)``.  Defaults to ``0.2``.
        val_size (float): Fraction of the *training portion* reserved for
            validation.  Must be in ``(0, 1)``.  Defaults to ``0.1``.
        group_column (str | None): Column name used as the group key when
            ``method="grouped"``.  Ignored for all other methods.

    Returns:
        SplitConfig: Validated split specification.
    """

    model_config = ConfigDict(extra="forbid")

    method: str = "stratified"
    test_size: float = 0.2
    val_size: float = 0.1
    group_column: str | None = None

    @field_validator("method")
    @classmethod
    def valid_method(cls, v: str) -> str:
        allowed = {"stratified", "random", "temporal", "grouped"}
        if v not in allowed:
            raise ValueError(f"method must be one of {allowed}, got {v!r}")
        return v

    @field_validator("test_size", "val_size")
    @classmethod
    def valid_fraction(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError(f"Must be in (0, 1), got {v}")
        return v


# ── EDA ───────────────────────────────────────────────────────────────────────


class EDAConfig(BaseModel):
    """Controls which EDA stages run as part of the pipeline.

    All stages default to enabled for a thorough first-pass report.
    Set ``multivariate=False`` when the feature count exceeds 200 to
    avoid generating an unreadable correlation matrix.

    Args:
        enabled (bool): Master switch.  When ``False``, the entire EDA
            stage is skipped.  Defaults to ``True``.
        univariate (bool): Run per-column summary statistics and distribution
            plots.  Defaults to ``True``.
        bivariate (bool): Run target-vs-feature analysis (IV, event rates).
            Defaults to ``True``.
        multivariate (bool): Run pairwise correlation matrix and heatmap.
            Defaults to ``False`` — expensive on 500+ columns.
        high_missing_threshold (float): Missing-value rate above which a
            feature triggers a ``HIGH_MISSING`` alert in the EDA Summary
            section.  Defaults to ``0.30``.
        near_zero_variance_threshold (float): Variance below which a numeric
            feature is flagged ``near_zero_variance_flag`` in the Univariate
            EDA numeric summary, and included in
            ``EDAReport.recommended_drops``/``recommended_drops_with_reasons``.
            Defaults to ``0.01``.
        skewness_alert_threshold (float): Absolute skewness at-or-above
            which a numeric feature triggers a ``SKEWED`` alert in the EDA
            Summary section.  Defaults to ``1.0``.
        zero_pct_alert_threshold (float): Zero-value rate at-or-above which
            a numeric feature triggers a ``ZEROS`` alert.  Defaults to
            ``0.05``.
        imbalance_alert_threshold (float): Entropy-based imbalance score
            (0=uniform, 1=dominated) at-or-above which a categorical
            feature triggers an ``IMBALANCED`` alert.  Defaults to ``0.5``.
        extreme_values_n (int): Number of smallest/largest raw values
            captured per numeric feature in the EDA Summary section.
            Defaults to ``5``.
        sample_n_rows (int): Number of rows shown per head/tail in the
            Sample section. Defaults to ``10``.
        duplicate_rows_max_display (int): Cap on the number of duplicate
            rows rendered in the Duplicate Rows section. Defaults to ``50``.
        missing_matrix_max_rows (int): Row subsample cap for the
            missing-values nullity matrix visualisation. Defaults to ``500``.
        interaction_max_numeric_cols (int): Cap on the number of numeric
            columns considered in the Interactions section, bounding the
            O(n^2) pair count. Defaults to ``8``.
        interaction_hexbin_row_threshold (int): Row count above which an
            Interactions pair plot switches from a scatter to a hexbin-style
            2D density plot. Defaults to ``2000``.
        text_analysis_top_n_words (int): Number of top word-frequency
            entries captured per categorical/text feature. Defaults to ``20``.
        text_analysis_top_n_chars (int): Number of top character-frequency
            entries captured per categorical/text feature. Defaults to ``20``.
        include_sample_rows (bool): Compliance opt-in — when ``False``
            (default), the Sample section (raw first/last rows) is omitted
            entirely from the report, since it exposes row-level data.
        include_duplicate_row_content (bool): Compliance opt-in — when
            ``False`` (default), the Duplicate Rows section shows only the
            existing duplicate *count*, never the actual row content.
        include_text_sample_values (bool): Compliance opt-in — when
            ``False`` (default), the categorical Words/Characters tab omits
            "Sample first 5 values" (raw value exposure); aggregate
            word/character frequency stats are unaffected by this flag.
        chart (bool): Master switch for every native chart in
            ``ModelCard.to_excel()`` — IV bar chart, correlation heatmap,
            missing-% bar chart, decile combo chart, interactions scatter
            charts, and the per-feature numeric/categorical distribution
            charts. When ``False``, sheets whose only content is a chart
            (Interactions, Numeric Charts, Categorical Charts) are omitted
            entirely; sheets with a standalone data table (IV Ranking,
            Correlations, Missing, Decile Table) keep their table and only
            drop the chart/conditional-format. Defaults to ``True``.
        chart_clip_lower_pct (float): Lower-tail fraction clipped before
            building the per-feature numeric distribution histogram, so a
            few extreme values don't stretch the x-axis and flatten the
            real distribution into one bin. Chart rendering only — never
            applied to modelling data. Defaults to ``0.01``.
        chart_clip_upper_pct (float): Upper-tail fraction clipped before
            building the per-feature numeric distribution histogram.
            Chart rendering only. Defaults to ``0.01``.

    Returns:
        EDAConfig: Validated EDA stage specification.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    univariate: bool = True
    bivariate: bool = True
    multivariate: bool = False
    high_missing_threshold: float = 0.30
    near_zero_variance_threshold: float = 0.01
    skewness_alert_threshold: float = 1.0
    zero_pct_alert_threshold: float = 0.05
    imbalance_alert_threshold: float = 0.5
    extreme_values_n: int = 5
    sample_n_rows: int = 10
    duplicate_rows_max_display: int = 50
    missing_matrix_max_rows: int = 500
    interaction_max_numeric_cols: int = 8
    interaction_hexbin_row_threshold: int = 2000
    text_analysis_top_n_words: int = 20
    text_analysis_top_n_chars: int = 20
    include_sample_rows: bool = False
    include_duplicate_row_content: bool = False
    include_text_sample_values: bool = False
    chart: bool = True
    chart_clip_lower_pct: float = 0.01
    chart_clip_upper_pct: float = 0.01

    @field_validator("skewness_alert_threshold")
    @classmethod
    def valid_skewness_threshold(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"skewness_alert_threshold must be >= 0, got {v}")
        return v

    @field_validator(
        "high_missing_threshold", "zero_pct_alert_threshold", "imbalance_alert_threshold"
    )
    @classmethod
    def valid_eda_threshold_01(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError(f"Threshold must be in (0, 1), got {v}")
        return v

    @field_validator("near_zero_variance_threshold")
    @classmethod
    def valid_near_zero_variance_threshold(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(f"near_zero_variance_threshold must be >= 0, got {v}")
        return v

    @field_validator("extreme_values_n")
    @classmethod
    def positive_extreme_values_n(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"extreme_values_n must be >= 1, got {v}")
        return v

    @field_validator(
        "sample_n_rows",
        "duplicate_rows_max_display",
        "missing_matrix_max_rows",
        "interaction_max_numeric_cols",
        "interaction_hexbin_row_threshold",
        "text_analysis_top_n_words",
        "text_analysis_top_n_chars",
    )
    @classmethod
    def positive_count(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"value must be >= 1, got {v}")
        return v

    @field_validator("chart_clip_lower_pct", "chart_clip_upper_pct")
    @classmethod
    def valid_chart_clip_pct(cls, v: float) -> float:
        if not (0.0 <= v < 0.5):
            raise ValueError(f"chart clip fraction must be in [0.0, 0.5), got {v!r}")
        return v


# ── Features ──────────────────────────────────────────────────────────────────


class ImputerConfig(BaseModel):
    """Configures the SmartImputer for handling missing values.

    Args:
        numeric_strategy (str): Strategy for numeric columns.  One of
            ``"auto"`` (skewness-based: median if \\|skew\\| ≥ 0.5, else mean),
            ``"mean"``, ``"median"``, ``"constant"``.  Defaults to ``"auto"``.
        categorical_strategy (str): Strategy for non-numeric columns.  One
            of ``"most_frequent"``, ``"constant"``.  Defaults to
            ``"most_frequent"``.
        fill_value (float | str | None): Constant used when either strategy
            is ``"constant"``.  Ignored otherwise.
        add_missing_indicator (bool): When ``True``, appends a binary
            ``{col}_was_missing`` column for every column that had nulls in
            the training set.  Defaults to ``False``.
        column_strategies (dict[str, str]): Per-column strategy override,
            keyed by column name.  A column listed here uses this strategy
            instead of ``numeric_strategy``/``categorical_strategy``.  Valid
            values are ``"auto"``, ``"mean"``, ``"median"``, ``"constant"``
            (numeric) or ``"most_frequent"``, ``"constant"`` (categorical) —
            dtype-appropriateness is the caller's responsibility, since this
            config has no access to the data's column types.  Defaults to
            ``{}`` (no overrides).
        column_fill_values (dict[str, Any]): Per-column constant fill value
            override, keyed by column name.  Only used for a column whose
            resolved strategy is ``"constant"``; falls back to ``fill_value``
            when absent.  Defaults to ``{}`` (no overrides).

    Returns:
        ImputerConfig: Validated imputer specification.
    """

    model_config = ConfigDict(extra="forbid")

    numeric_strategy: str = "auto"
    categorical_strategy: str = "most_frequent"
    fill_value: float | str | None = None
    add_missing_indicator: bool = False
    column_strategies: dict[str, str] = Field(default_factory=dict)
    column_fill_values: dict[str, Any] = Field(default_factory=dict)

    @field_validator("numeric_strategy")
    @classmethod
    def valid_numeric(cls, v: str) -> str:
        allowed = {"auto", "mean", "median", "constant"}
        if v not in allowed:
            raise ValueError(f"numeric_strategy must be one of {allowed}, got {v!r}")
        return v

    @field_validator("categorical_strategy")
    @classmethod
    def valid_categorical(cls, v: str) -> str:
        allowed = {"most_frequent", "constant"}
        if v not in allowed:
            raise ValueError(f"categorical_strategy must be one of {allowed}, got {v!r}")
        return v

    @field_validator("column_strategies")
    @classmethod
    def valid_column_strategies(cls, v: dict[str, str]) -> dict[str, str]:
        allowed = {"auto", "mean", "median", "constant", "most_frequent"}
        invalid = {col: strategy for col, strategy in v.items() if strategy not in allowed}
        if invalid:
            raise ValueError(
                f"column_strategies has invalid values: {invalid} — must be one of {allowed}"
            )
        return v


class EncoderConfig(BaseModel):
    """Configures categorical encoding.

    Args:
        strategy (str): Encoding strategy.  One of ``"ordinal"`` (default —
            safe for tree models), ``"onehot"``, ``"target"`` (target-mean
            encoding, requires target during fit).
        max_categories (int): Columns with more than this many unique values
            are treated as high-cardinality and dropped by the
            ``CardinalitySelector`` instead of being encoded.  Defaults to
            ``50``.

    Returns:
        EncoderConfig: Validated encoder specification.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: str = "ordinal"
    max_categories: int = 50

    @field_validator("strategy")
    @classmethod
    def valid_strategy(cls, v: str) -> str:
        allowed = {"ordinal", "onehot", "target"}
        if v not in allowed:
            raise ValueError(f"strategy must be one of {allowed}, got {v!r}")
        return v


class ScalerConfig(BaseModel):
    """Configures numeric scaling applied after imputation.

    Scaling is not required for tree-based models (xgboost, lightgbm,
    random_forest) and defaults to ``"none"``.  Set to ``"standard"`` or
    ``"robust"`` when using logistic regression.

    Args:
        strategy (str): Scaling strategy.  One of ``"none"`` (default),
            ``"standard"`` (zero mean, unit variance), ``"minmax"``
            (scale to [0, 1]), ``"robust"`` (IQR-based, outlier-resistant).

    Returns:
        ScalerConfig: Validated scaler specification.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: str = "none"

    @field_validator("strategy")
    @classmethod
    def valid_strategy(cls, v: str) -> str:
        allowed = {"none", "standard", "minmax", "robust"}
        if v not in allowed:
            raise ValueError(f"strategy must be one of {allowed}, got {v!r}")
        return v


class DistributionConfig(BaseModel):
    """Configures distribution-normalising transforms for skewed numeric columns.

    Runs after winsorization, imputation, and categorical encoding, and
    before scaling — normalising skew is most effective on already-imputed
    data, and the scaler should see the transformed distribution rather than
    the raw skewed one.

    Args:
        strategy (str): Transform to apply. One of ``"none"`` (default),
            ``"log"`` (requires every value strictly positive), ``"log1p"``
            (requires every value > -1), or ``"yeo_johnson"`` (handles zero
            and negative values — the right default for skewed banking
            numerics like balance or income that can go negative).

    Returns:
        DistributionConfig: Validated distribution-transform specification.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: str = "none"

    @field_validator("strategy")
    @classmethod
    def valid_strategy(cls, v: str) -> str:
        allowed = {"none", "log", "log1p", "yeo_johnson"}
        if v not in allowed:
            raise ValueError(f"strategy must be one of {allowed}, got {v!r}")
        return v


class WinsorizerConfig(BaseModel):
    """Configures outlier capping applied before imputation.

    Runs ``WinsorizationTransformer`` on raw numeric columns, before
    imputation, so percentile boundaries are computed from the true
    distribution rather than one already containing imputed fill values.
    Disabled by default — tree-based models are largely outlier-robust;
    enable for logistic regression or other scale-sensitive algorithms
    where extreme values disproportionately influence the fit.

    Args:
        enabled (bool): Whether to apply winsorization. Defaults to
            ``False``.
        lower_tail (float): Fraction of the lower tail to clip. Defaults
            to ``0.01`` (1st percentile), matching
            ``settings.winsorizer_lower_tail``.
        upper_tail (float): Fraction of the upper tail to clip. Defaults
            to ``0.01`` (99th percentile), matching
            ``settings.winsorizer_upper_tail``.

    Returns:
        WinsorizerConfig: Validated winsorizer specification.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    lower_tail: float = 0.01
    upper_tail: float = 0.01

    @field_validator("lower_tail", "upper_tail")
    @classmethod
    def valid_tail_fraction(cls, v: float) -> float:
        if not (0.0 <= v < 0.5):
            raise ValueError(f"tail fraction must be in [0.0, 0.5), got {v!r}")
        return v


class FeaturesConfig(BaseModel):
    """Groups all feature engineering stage configurations.

    Args:
        imputer (ImputerConfig): Missing value imputation settings.
        encoder (EncoderConfig): Categorical encoding settings.
        scaler (ScalerConfig): Numeric scaling settings.
        distribution (DistributionConfig): Distribution-normalising
            transform settings, applied after encoding and before scaling.
        winsorizer (WinsorizerConfig): Outlier-capping settings, applied
            before imputation.

    Returns:
        FeaturesConfig: Validated feature engineering specification.
    """

    model_config = ConfigDict(extra="forbid")

    imputer: ImputerConfig = Field(default_factory=ImputerConfig)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    scaler: ScalerConfig = Field(default_factory=ScalerConfig)
    distribution: DistributionConfig = Field(default_factory=DistributionConfig)
    winsorizer: WinsorizerConfig = Field(default_factory=WinsorizerConfig)


# ── Target ────────────────────────────────────────────────────────────────────


class ImbalanceConfig(BaseModel):
    """Controls how class imbalance is handled during training.

    Applies only to classification tasks.  ``"class_weight"`` is the
    default and safest option — it reweights the loss without modifying the
    training set.  SMOTE and resampling strategies modify row counts.

    Args:
        strategy (str): Imbalance correction strategy.  One of
            ``"class_weight"`` (default), ``"smote"``, ``"undersample"``,
            ``"oversample"``, ``"none"``.
        sampling_strategy (float | str): Ratio or strategy string forwarded
            to the underlying sampler.  Defaults to ``"auto"``.

    Returns:
        ImbalanceConfig: Validated imbalance specification.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: str = "class_weight"
    sampling_strategy: float | str = "auto"

    @field_validator("strategy")
    @classmethod
    def valid_strategy(cls, v: str) -> str:
        allowed = {"class_weight", "smote", "undersample", "oversample", "none"}
        if v not in allowed:
            raise ValueError(f"strategy must be one of {allowed}, got {v!r}")
        return v


class TargetConfig(BaseModel):
    """Configures target variable treatment before model training.

    Args:
        imbalance (ImbalanceConfig): Class imbalance correction settings.
            Applied to classification tasks only.
        binarize_threshold (float | None): When set, a continuous target
            column is converted to a binary label: ``1`` if value >=
            ``binarize_threshold``, else ``0``.  ``None`` means the target
            column is used as-is.

    Returns:
        TargetConfig: Validated target treatment specification.
    """

    model_config = ConfigDict(extra="forbid")

    imbalance: ImbalanceConfig = Field(default_factory=ImbalanceConfig)
    binarize_threshold: float | None = None


# ── Selection ─────────────────────────────────────────────────────────────────


class SelectionConfig(BaseModel):
    """Controls which feature selection steps run and their thresholds.

    Steps run in this fixed order: null rate → constant → quasi-constant →
    cardinality → leakage check → correlation → IV → VIF (optional).  Each
    step removes features from the working set before the next step sees
    them.

    All thresholds here override the global ``settings`` defaults for this
    experiment only.

    Args:
        remove_high_null (bool): Drop columns whose missing-value rate
            exceeds ``null_rate_threshold``.  Runs first, before constant
            detection.  Defaults to ``True``.
        null_rate_threshold (float): Missing-value rate above which a
            column is dropped.  Defaults to ``0.8``.
        remove_constant (bool): Drop columns whose variance is zero.
            Defaults to ``True``.
        remove_quasi_constant (bool): Drop columns where one value dominates
            above ``quasi_constant_threshold``.  Defaults to ``True``.
        quasi_constant_threshold (float): Frequency threshold for
            quasi-constant detection.  Defaults to ``0.99``.
        remove_high_cardinality (bool): Drop categorical columns with more
            than ``cardinality_threshold`` unique values.  Defaults to ``True``.
        cardinality_threshold (int): Unique-value cutoff for cardinality
            removal.  Defaults to ``50``.
        leakage_check (bool): Flag and optionally remove features whose
            Pearson correlation with the target exceeds
            ``leakage_threshold``.  Defaults to ``True`` (always runs).
        leakage_threshold (float): Correlation above which a feature is
            flagged as a potential leakage risk.  Defaults to ``0.95``.
        remove_high_correlation (bool): Drop one column from each highly
            correlated pair.  Defaults to ``True``.
        correlation_threshold (float): Pairwise Pearson r above which one
            column is dropped.  Defaults to ``0.85``.
        iv_threshold (float): Information Value below which a feature is
            dropped.  Defaults to ``0.02`` (weak predictors).
        vif_enabled (bool): Run Variance Inflation Factor analysis.
            Expensive on 500+ columns — disabled by default.
        vif_threshold (float): VIF above which a feature is flagged for
            multicollinearity.  Defaults to ``10.0``.

    Returns:
        SelectionConfig: Validated feature selection specification.
    """

    model_config = ConfigDict(extra="forbid")

    remove_high_null: bool = True
    null_rate_threshold: float = 0.8
    remove_constant: bool = True
    remove_quasi_constant: bool = True
    quasi_constant_threshold: float = 0.99
    remove_high_cardinality: bool = True
    cardinality_threshold: int = 50
    leakage_check: bool = True
    leakage_threshold: float = 0.95
    remove_high_correlation: bool = True
    correlation_threshold: float = 0.85
    iv_threshold: float = 0.02
    vif_enabled: bool = False
    vif_threshold: float = 10.0

    @field_validator(
        "quasi_constant_threshold",
        "leakage_threshold",
        "correlation_threshold",
        "null_rate_threshold",
    )
    @classmethod
    def valid_threshold_01(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError(f"Threshold must be in (0, 1), got {v}")
        return v

    @field_validator("iv_threshold")
    @classmethod
    def valid_iv(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"iv_threshold must be >= 0, got {v}")
        return v


# ── Model ─────────────────────────────────────────────────────────────────────


class ModelConfig(BaseModel):
    """Specifies the ML task, algorithm, and any parameter overrides.

    The ``task`` field is required — it determines the evaluation metrics,
    the model wrapper class, and the available algorithms.  All other fields
    have defaults.  Parameters in ``params`` are merged on top of dscompanion's
    built-in defaults for the chosen algorithm, so only overrides need to
    be specified.

    Args:
        task (str): ML task type.  One of ``"classification"``,
            ``"regression"``, ``"clustering"``.  Required.
        algorithm (str): Algorithm identifier within the task.  Validated
            against ``ModelFactory.SUPPORTED_ALGORITHMS[task]`` — the single
            source of truth shared with ``ModelFactory.build()`` and
            ``Leaderboard``, so this list is never duplicated/out of sync
            here.
        params (dict): Parameter overrides applied on top of dscompanion's
            built-in defaults for this algorithm.  Empty dict = use all
            defaults.

    Returns:
        ModelConfig: Validated model specification.
    """

    model_config = ConfigDict(extra="forbid")

    task: str
    algorithm: str = "xgboost"
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task")
    @classmethod
    def valid_task(cls, v: str) -> str:
        allowed = {"classification", "regression", "clustering"}
        v = v.lower()
        if v not in allowed:
            raise ValueError(f"task must be one of {allowed}, got {v!r}")
        return v

    @field_validator("algorithm")
    @classmethod
    def valid_algorithm(cls, v: str, info: ValidationInfo) -> str:
        from dscompanion.models import ModelFactory

        v = v.lower()
        task = info.data.get("task")
        if task is None:
            # task itself already failed validation — let that error surface,
            # don't raise a second, confusing one here.
            return v
        allowed = ModelFactory.SUPPORTED_ALGORITHMS.get(task, [])
        if v not in allowed:
            raise ValueError(f"algorithm must be one of {allowed} for task={task!r}, got {v!r}")
        return v


# ── Tuning ────────────────────────────────────────────────────────────────────


class TuningConfig(BaseModel):
    """Controls hyperparameter optimisation via Optuna.

    Disabled by default so the baseline pipeline runs fast.  Enable for
    production runs after the baseline has validated data quality and
    feature selection.

    Args:
        enabled (bool): When ``False`` (default), hyperparameter tuning is
            skipped and the model trains with default or overridden params.
        n_trials (int): Number of Optuna trials when enabled.  Defaults to
            ``50`` — enough for a good baseline sweep.  Use 100–200 for
            production.
        metric (str): Metric key to optimise.  Must match a key returned by
            the model's ``evaluate()`` method.  Defaults to ``"roc_auc"``.
        direction (str): ``"maximize"`` (default) or ``"minimize"``.
        backend (str): Tuning backend.  Only ``"optuna"`` is supported
            (optuna==3.4.0 is in the approved package list).

    Returns:
        TuningConfig: Validated tuning specification.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    n_trials: int = 50
    metric: str = "roc_auc"
    direction: str = "maximize"
    backend: str = "optuna"

    @field_validator("direction")
    @classmethod
    def valid_direction(cls, v: str) -> str:
        allowed = {"maximize", "minimize"}
        if v not in allowed:
            raise ValueError(f"direction must be one of {allowed}, got {v!r}")
        return v

    @field_validator("backend")
    @classmethod
    def valid_backend(cls, v: str) -> str:
        if v != "optuna":
            raise ValueError("Only 'optuna' is supported (optuna==3.4.0 is approved)")
        return v

    @field_validator("n_trials")
    @classmethod
    def positive_trials(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"n_trials must be >= 1, got {v}")
        return v


# ── Leaderboard ───────────────────────────────────────────────────────────────


class LeaderboardConfig(BaseModel):
    """Controls multi-algorithm comparison via ``dscompanion.leaderboard.Leaderboard``.

    Disabled by default — the pipeline trains exactly ``model.algorithm``,
    unchanged. When enabled, ``PipelineRunner`` trains every algorithm
    supported for ``model.task`` (or the ``include``/``exclude`` subset),
    and the top-ranked candidate replaces ``model.algorithm`` for the rest
    of the pipeline (tuning, calibration, SHAP, model card) — same
    "compare, then finalise" workflow as PyCaret's ``compare_models()`` /
    H2O AutoML's leaderboard.

    Args:
        enabled (bool): When ``False`` (default), leaderboard comparison is
            skipped entirely and the pipeline behaves exactly as it did
            before this config existed.
        include (list[str], optional): Restrict comparison to these
            algorithm names. Each must belong to
            ``ModelFactory.SUPPORTED_ALGORITHMS[model.task]``. Defaults to
            ``None`` (every supported algorithm for ``model.task``).
        exclude (list[str], optional): Algorithm names to skip. Ignored
            when ``include`` is supplied. Defaults to ``None``.
        sort_metric (str, optional): Metric column to rank by. Defaults to
            ``None``, which resolves to ``Leaderboard``'s own default
            (``settings.classification_metrics[0]``, i.e. ``"roc_auc"``).
        eval_split (str, optional): ``DataSplit`` partition to evaluate and
            rank on. Defaults to ``None``, which resolves to ``Leaderboard``'s
            own ``val``-else-``test`` fallback.

    Returns:
        LeaderboardConfig: Validated leaderboard specification.

    Raises:
        pydantic.ValidationError: If ``enabled=True`` while ``model.task``
            is not ``"classification"`` (only task currently supported by
            ``Leaderboard``), or if ``include``/``exclude`` names an
            algorithm not in ``ModelFactory.SUPPORTED_ALGORITHMS``. Both
            checks run at the ``PipelineConfig`` level — see
            ``PipelineConfig.validate_leaderboard_compatibility``.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    include: list[str] | None = None
    exclude: list[str] | None = None
    sort_metric: str | None = None
    eval_split: str | None = None


# ── Explain ───────────────────────────────────────────────────────────────────


class ExplainConfig(BaseModel):
    """Controls model explainability outputs (SHAP and LIME).

    SHAP uses TreeExplainer with subsampling, making it feasible on 2M-row
    datasets — but it is **off by default** (temporarily, see below).  LIME
    is also off by default — it is slower and less stable on large feature
    sets.

    Args:
        shap_enabled (bool): Generate SHAP feature importance summary.
            Uses ``shap.TreeExplainer`` for tree models and
            ``shap.LinearExplainer`` for linear models.  Defaults to
            ``False`` — SHAP computation is relatively expensive, so it's
            opt-in rather than run on every pipeline call.
        shap_sample_size (int): Number of rows sampled from the test set for
            SHAP computation.  Reduces memory and runtime on large datasets.
            Defaults to ``5000``.
        shap_top_n (int): Number of top features to include in the SHAP
            summary plot.  Defaults to ``20``.
        lime_enabled (bool): Generate LIME local explanations for a sample
            of test instances.  Defaults to ``False``.
        permutation_enabled (bool): Generate permutation-based feature
            importance via ``PermutationImportanceAnalyser``. Model-agnostic
            and needs no ``shap.TreeExplainer`` — a lightweight alternative
            (or addition) to SHAP. Defaults to ``False``, independent of
            ``shap_enabled``.
        permutation_n_repeats (int): Number of times each feature is
            shuffled and re-scored.  Defaults to ``10``.
        permutation_sample_size (int): Max rows sampled from the test set
            for permutation importance.  Defaults to ``5000``.
        permutation_top_n (int): Number of top features to include in the
            permutation importance summary plot.  Defaults to ``20``.

    Returns:
        ExplainConfig: Validated explainability specification.
    """

    model_config = ConfigDict(extra="forbid")

    shap_enabled: bool = False
    shap_sample_size: int = 5000
    shap_top_n: int = 20
    lime_enabled: bool = False
    permutation_enabled: bool = False
    permutation_n_repeats: int = 10
    permutation_sample_size: int = 5000
    permutation_top_n: int = 20

    @field_validator(
        "shap_sample_size",
        "shap_top_n",
        "permutation_n_repeats",
        "permutation_sample_size",
        "permutation_top_n",
    )
    @classmethod
    def positive_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"Must be >= 1, got {v}")
        return v


# ── Reporting ─────────────────────────────────────────────────────────────────


class ReportingConfig(BaseModel):
    """Controls output artefacts: run tracking, HTML report, and model card content.

    Args:
        output_dir (str): Root directory this run's artifacts are organized
            under. ``PipelineRunner`` creates ``<output_dir>/<run_id>/``
            (``run_id`` a ``yyyymmdd_hhmmss`` timestamp — see
            ``PipelineRunResult.run_id``/``.run_dir``) with ``model/``,
            ``reports/``, ``logs/``, and ``eda/`` subdirectories — declared
            once here, not per-artifact. On Databricks, use an
            ``abfss://`` path or ``/dbfs/`` mount. Defaults to
            ``"./reports"``.
        html_report (bool): Also write the HTML model-card report to
            ``<run_dir>/reports/`` (and log it as a tracked artifact). The
            Excel model card is written unconditionally regardless of this
            flag (see ``PipelineRunResult.excel_report_path``) — this flag
            only controls the *additional* ``.to_html()`` file write.
            Defaults to ``False``.
        decile_table (bool): Include a 10-decile lift table in the report.
            Classification only. Defaults to ``True``.

    Returns:
        ReportingConfig: Validated reporting specification.
    """

    model_config = ConfigDict(extra="forbid")

    output_dir: str = "./reports"
    html_report: bool = False
    decile_table: bool = True


# ── Root ──────────────────────────────────────────────────────────────────────


class PipelineConfig(BaseModel):
    """Root configuration object for a full dscompanion experiment.

    Loaded from a YAML file by the scientist and passed directly to
    ``PipelineRunner``.  Validates all nested configs at construction time,
    rejecting typos and invalid values before a single line of the pipeline
    runs.

    Minimum viable YAML (everything else defaults)::

        name: credit_risk_v1
        data:
          path: abfss://container@account.dfs.core.windows.net/features/credit.parquet
          target: default_flag
        model:
          task: classification

    Args:
        name (str): Experiment name.  Used as the run name and report
            title.  Required.
        version (str): Experiment version string.  Defaults to ``"1.0"``.
        description (str): Free-text description of what this experiment
            tests.  Defaults to empty string.
        owner (str): Scientist name or UCIC.  Included in the model card.
            Defaults to empty string.
        data (DataConfig): Data source specification.  Required.
        split (SplitConfig): Train/val/test split configuration.
        eda (EDAConfig): EDA stage configuration.
        features (FeaturesConfig): Feature engineering configuration.
        feature_recipes (dict[str, ColumnRecipe] | None): Per-column ordered transformation
            recipes (see ``dscompanion.features.transform_chain.ColumnRecipe``). When set, overrides
            ``features`` entirely for feature processing — a ``FeatureTransformChain`` built from
            these recipes replaces the ``SmartImputer``/``FeatureProcessingPipeline`` stage.
            Defaults to ``None`` (today's global per-dtype ``features.*`` strategies apply,
            unchanged).
        target (TargetConfig): Target variable treatment configuration.
        selection (SelectionConfig): Feature selection configuration.
        model (ModelConfig): Model task and algorithm configuration.  Required.
        tuning (TuningConfig): Hyperparameter tuning configuration.
        leaderboard (LeaderboardConfig): Multi-algorithm comparison
            configuration.  When enabled, the top-ranked algorithm replaces
            ``model.algorithm`` for the rest of the pipeline.
        explain (ExplainConfig): Explainability outputs configuration.
        reporting (ReportingConfig): Run tracking and report output configuration.

    Returns:
        PipelineConfig: Fully validated pipeline configuration ready to
        pass to ``PipelineRunner``.

    Raises:
        pydantic.ValidationError: If any required field is missing or any
            value fails validation.  The error message names every failing
            field so the scientist can fix all issues in one pass.  Includes
            ``leaderboard``/``model.task`` cross-field checks — see
            ``validate_leaderboard_compatibility``.
    """

    model_config = ConfigDict(extra="forbid")

    # Experiment identity
    name: str
    version: str = "1.0"
    description: str = ""
    owner: str = ""

    # Pipeline stages
    data: DataConfig
    split: SplitConfig = Field(default_factory=SplitConfig)
    eda: EDAConfig = Field(default_factory=EDAConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    feature_recipes: dict[str, ColumnRecipe] | None = None
    target: TargetConfig = Field(default_factory=TargetConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    model: ModelConfig
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    leaderboard: LeaderboardConfig = Field(default_factory=LeaderboardConfig)
    explain: ExplainConfig = Field(default_factory=ExplainConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    @model_validator(mode="after")
    def validate_leaderboard_compatibility(self) -> "PipelineConfig":
        if not self.leaderboard.enabled:
            return self

        from dscompanion.models import ModelFactory

        if self.model.task != "classification":
            raise ValueError(
                "leaderboard.enabled=True currently requires model.task='classification', "
                f"got {self.model.task!r}"
            )
        allowed = set(ModelFactory.SUPPORTED_ALGORITHMS["classification"])
        for field_name in ("include", "exclude"):
            names = getattr(self.leaderboard, field_name)
            if names:
                unknown = set(names) - allowed
                if unknown:
                    raise ValueError(
                        f"leaderboard.{field_name} has unknown algorithm(s): {sorted(unknown)}"
                    )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load and validate a PipelineConfig from a YAML file.

        Reads the YAML, parses it into a dict, and passes it through pydantic
        validation.  Any missing required fields or invalid values raise
        ``pydantic.ValidationError`` with a human-readable message before the
        pipeline starts.

        Args:
            path (str | Path): Path to the YAML config file.

        Returns:
            PipelineConfig: Fully validated pipeline configuration.

        Raises:
            FileNotFoundError: If the file does not exist at ``path``.
            yaml.YAMLError: If the file is not valid YAML.
            pydantic.ValidationError: If the config fails validation.
        """
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh)
        return cls(**raw)

    def to_yaml(self, path: str | Path) -> None:
        """Serialise this config to a YAML file.

        Writes all fields including defaults so the saved file is a complete,
        self-documenting record of the exact configuration that was run.

        Args:
            path (str | Path): Destination path for the YAML file.  Parent
                directory must exist.

        Returns:
            None
        """
        with open(path, "w") as fh:
            yaml.dump(
                self.model_dump(),
                fh,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
