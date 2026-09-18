"""Central configuration — single source of truth for all defaults.

All modules import from here. Never hardcode a threshold elsewhere.
"""

from __future__ import annotations

import pydantic as _pydantic

if int(_pydantic.VERSION.split(".")[0]) >= 2:
    try:
        from pydantic_settings import BaseSettings
    except ImportError:
        from pydantic.v1 import BaseSettings  # pydantic v2 bundles v1 compat layer
else:
    from pydantic import BaseSettings  # pydantic v1.x
from pydantic import Field

__all__ = ["DSCompanionConfig", "settings"]


class DSCompanionConfig(BaseSettings):
    """Holds all global configuration defaults for the dscompanion package and
    exposes them as typed attributes that can be overridden at runtime via
    environment variables or a ``.env`` file, without modifying source code.

    Every field is readable as a plain Python attribute (e.g.
    ``settings.random_state``).  No side-effects occur at import time beyond
    reading environment variables and the optional ``.env`` file from the
    current working directory.

    Args:
        default_test_size (float): Fraction of total data reserved for the
            test split.  Must be in ``(0, 1)``.  Defaults to ``0.2``.
        default_val_size (float): Fraction of the training portion reserved
            for the validation split.  Must be in ``(0, 1)``.  Defaults to
            ``0.1``.
        random_state (int): Global random seed used wherever reproducible
            randomness is required.  Defaults to ``42``.
        run_id_timezone (str): IANA timezone name used to timestamp
            ``PipelineRunner``'s ``run_id`` and run directory.  Defaults to
            ``"UTC"``.
        high_cardinality_threshold (int): Number of unique values above which
            a categorical column is considered high-cardinality during EDA.
            Defaults to ``50``.
        correlation_threshold (float): Pearson correlation coefficient above
            which one of a correlated pair of columns is dropped.  Defaults
            to ``0.85``.
        near_zero_variance_threshold (float): Variance below which a numeric
            column is treated as effectively constant.  Defaults to ``0.01``.
        quasi_constant_threshold (float): Relative frequency of the most
            common value above which a column is considered quasi-constant.
            Defaults to ``0.99``.
        iv_threshold (float): Information Value below which a feature is
            dropped during feature selection.  Defaults to ``0.02``.
        vif_threshold (float): Variance Inflation Factor above which a
            feature is flagged for multicollinearity.  Defaults to ``10.0``.
        target_leakage_correlation_threshold (float): Pearson correlation
            with the target above which a feature is flagged as a potential
            leakage risk.  Defaults to ``0.95``.
        classification_metrics (list[str]): Ordered list of metric keys
            computed for classification models.  Defaults to
            ``["roc_auc", "ks_statistic", "gini", "f1", "log_loss", "psi"]``.
        regression_metrics (list[str]): Ordered list of metric keys computed
            for regression models.  Defaults to
            ``["rmse", "mae", "r2", "mape"]``.
        clustering_metrics (list[str]): Ordered list of metric keys computed
            for clustering models.  Defaults to
            ``["silhouette", "davies_bouldin", "inertia"]``.
        default_n_trials (int): Default number of hyperparameter optimisation
            trials for the tuning module.  Defaults to ``100``.
        default_cv_folds (int): Default number of cross-validation folds used
            during tuning.  Defaults to ``5``.
        max_eda_rows (int): Maximum number of rows loaded into memory for
            EDA distribution plots to guard against OOM errors on large
            datasets.  Defaults to ``100_000``.

    Returns:
        DSCompanionConfig: A fully validated settings instance.  All values are
        overridable via environment variables prefixed with ``DSCOMPANION_``
        (e.g. ``DSCOMPANION_RANDOM_STATE=0``) or via a ``.env`` file in the
        current working directory.  Unknown environment variables are
        silently ignored.
    """

    # ── Split ────────────────────────────────────────────────────────────────
    default_test_size: float = Field(0.2, description="Fraction of data for test set")
    default_val_size: float = Field(0.1, description="Fraction of train for validation")
    random_state: int = 42

    # ── Run identity ─────────────────────────────────────────────────────────
    run_id_timezone: str = Field(
        "UTC",
        description=(
            "IANA timezone name used to timestamp PipelineRunner's run_id "
            "(yyyymmdd_hhmmss) and run directory. Defaults to UTC so run "
            "folders read consistently regardless of which machine or "
            "region generated them. Override via DSCOMPANION_RUN_ID_TIMEZONE."
        ),
    )

    # ── EDA ─────────────────────────────────────────────────────────────────
    high_cardinality_threshold: int = Field(
        50, description="Unique values above which a column is high-cardinality"
    )
    correlation_threshold: float = Field(
        0.85, description="Pearson r above which one column is dropped"
    )
    near_zero_variance_threshold: float = Field(
        0.01, description="Variance below which column is constant"
    )
    quasi_constant_threshold: float = Field(
        0.99, description="Single value frequency above which column is quasi-constant"
    )

    # ── Feature selection ────────────────────────────────────────────────────
    iv_threshold: float = Field(0.02, description="IV below this → feature dropped")
    vif_threshold: float = Field(10.0, description="VIF above this → multicollinearity flag")
    target_leakage_correlation_threshold: float = Field(0.95)
    null_rate_threshold: float = Field(
        0.8, description="Missing-value rate above which NullRateSelector drops a column"
    )

    # ── Calibration ──────────────────────────────────────────────────────────
    calibration_ece_bins: int = Field(
        10,
        description=(
            "Equal-width probability bins for Expected Calibration Error in Calibrator.ece()"
        ),
    )

    # ── Clustering ───────────────────────────────────────────────────────────
    silhouette_sample_size: int = Field(
        5000,
        description=(
            "Max rows sampled for silhouette_score in ClusteringModel.evaluate() — prevents OOM "
            "on large datasets"
        ),
    )

    # ── Model base ───────────────────────────────────────────────────────────
    score_dist_bins: int = Field(
        50,
        description=(
            "Number of histogram bins per class in ClassificationModel.score_distribution_plot()"
        ),
    )
    evaluate_round_precision: int = Field(
        6,
        description="Decimal places for metric values returned by BaseDSCompanionModel.evaluate()",
    )
    fit_time_round_precision: int = Field(
        3,
        description="Decimal places for fit duration logged after BaseDSCompanionModel.fit()",
    )

    # ── PSI ──────────────────────────────────────────────────────────────────
    psi_alert_threshold: float = Field(
        0.25, description="PSI above this value is flagged as significant distribution shift"
    )
    psi_feature_n_bins: int = Field(
        10,
        description=(
            "Number of equal-frequency bins used in feature_psi_table() for numeric columns"
        ),
    )
    csi_alert_threshold: float = Field(
        0.2,
        description=(
            "CSI (feature-level drift, frozen training reference) above this value is "
            "flagged as significant distribution shift — stricter than psi_alert_threshold "
            "since CSI compares raw business features, not model scores"
        ),
    )

    # ── Model metrics ────────────────────────────────────────────────────────
    classification_metrics: list[str] = ["roc_auc", "ks_statistic", "gini", "f1", "log_loss", "psi"]
    regression_metrics: list[str] = ["rmse", "mae", "r2", "mape"]
    clustering_metrics: list[str] = ["silhouette", "davies_bouldin", "inertia"]

    # ── Tuning ───────────────────────────────────────────────────────────────
    default_n_trials: int = 100
    default_cv_folds: int = 5

    # ── EDA memory guard ─────────────────────────────────────────────────────
    max_eda_rows: int = Field(100_000, description="Max rows loaded for distribution plots")

    # ── Multivariate EDA ─────────────────────────────────────────────────────
    vif_round_precision: int = Field(
        2, description="Decimal places for VIF values in output tables"
    )
    pca_default_components: int = Field(
        10,
        description="Default number of PCA components — used by pca_summary() (EDA diagnostic) "
        "and PCATransformer's n_components default (fitted pipeline transformer)",
    )
    pca_plot_max_components: int = Field(
        20, description="Maximum components rendered in the PCA scree plot"
    )

    # ── Bivariate EDA ────────────────────────────────────────────────────────
    default_iv_bins: int = Field(
        10, description="Default quantile bins for WoE/IV computation per numeric feature"
    )
    iv_round_precision: int = Field(6, description="Decimal places for IV values in output tables")
    corr_round_precision: int = Field(
        2, description="Decimal places for Pearson r and Cramér's V in output tables"
    )

    # ── EDA thresholds ───────────────────────────────────────────────────────
    high_missing_threshold: float = Field(
        0.30, description="Missing-value rate above which a column is flagged as high-missing"
    )
    iqr_multiplier: float = Field(
        1.5, description="IQR fence multiplier for outlier detection (Tukey rule)"
    )
    zscore_outlier_threshold: float = Field(
        3.0, description="Absolute z-score above which a value is counted as an outlier"
    )
    eda_histogram_bins: int = Field(
        40, description="Number of bins in EDA numeric distribution histograms"
    )
    eda_kde_points: int = Field(
        200, description="Number of x-axis points sampled when rendering a KDE curve overlay"
    )
    eda_top_categorical_values: int = Field(
        20, description="Top-N most frequent values shown in categorical EDA bar charts"
    )
    eda_chart_clip_lower_pct: float = Field(
        0.05,
        description=(
            "Lower-tail fraction clipped before building per-feature numeric "
            "distribution charts, when the user opts in via the 'exclude outliers' "
            "toggle (chart rendering only, never applied to modelling data)"
        ),
    )
    eda_chart_clip_upper_pct: float = Field(
        0.05,
        description=(
            "Upper-tail fraction clipped before building per-feature numeric "
            "distribution charts, when the user opts in via the 'exclude outliers' "
            "toggle (chart rendering only, never applied to modelling data)"
        ),
    )
    eda_skewness_alert_threshold: float = Field(
        1.0,
        description="Absolute skewness at-or-above which a numeric feature triggers a SKEWED alert",
    )
    eda_zero_pct_alert_threshold: float = Field(
        0.05,
        description="Zero-value rate at-or-above which a numeric feature triggers a ZEROS alert",
    )
    eda_imbalance_alert_threshold: float = Field(
        0.5,
        description=(
            "Entropy-based imbalance score (0=uniform, 1=dominated) at-or-above which a feature "
            "categorical triggers an IMBALANCED alert"
        ),
    )
    eda_extreme_values_n: int = Field(
        5,
        description=(
            "Number of smallest/largest raw values captured per numeric feature in "
            "UnivariateAnalyser.extreme_values()"
        ),
    )
    eda_sample_n_rows: int = Field(
        10, description="Rows shown per head/tail in the EDA Summary Sample section"
    )
    eda_duplicate_rows_max_display: int = Field(
        50, description="Cap on duplicate rows rendered in the EDA Summary Duplicate Rows section"
    )
    eda_missing_matrix_max_rows: int = Field(
        500, description="Row subsample cap for the EDA Summary missing-values nullity matrix"
    )
    eda_interaction_max_numeric_cols: int = Field(
        8, description="Cap on numeric columns considered in the EDA Summary Interactions section"
    )
    eda_interaction_hexbin_row_threshold: int = Field(
        2000, description="Row count above which an Interactions plot switches scatter to hexbin"
    )
    eda_text_analysis_top_n_words: int = Field(
        20, description="Top-N word-frequency entries captured per categorical/text feature"
    )
    eda_text_analysis_top_n_chars: int = Field(
        20, description="Top-N character-frequency entries captured per categorical/text feature"
    )
    eda_chart_export_dpi: int = Field(
        150,
        description="Pixel density for EDAReport.export_charts()'s PNG output (ignored for SVG)",
    )
    eda_bivariate_chart_top_n: int = Field(
        20,
        description=(
            "Max features EDAReport.export_charts() renders bivariate (target-rate-by-bin) "
            "charts for, ranked by Information Value"
        ),
    )

    # ── Feature engineering — AutoBinner ─────────────────────────────────────
    binner_n_bins: int = Field(
        10, description="Default number of bins for AutoBinner (quantile/uniform strategies)"
    )
    binner_min_samples_leaf: float = Field(
        0.05, description="Minimum fraction of training rows per leaf in tree-based AutoBinner"
    )
    binner_max_classes: int = Field(
        20,
        description=(
            "Unique target values at-or-below which tree AutoBinner uses a classifier; above → "
            "regressor"
        ),
    )

    # ── Feature engineering — SmartImputer ──────────────────────────────────
    imputer_skew_threshold: float = Field(
        0.5,
        description=(
            "Absolute skewness below which auto strategy uses mean; at-or-above uses median"
        ),
    )

    # ── Feature engineering — WoEEncoder ─────────────────────────────────────
    woe_clip_value: float = Field(
        4.0,
        description=(
            "Symmetric bound applied to WoE values to prevent ±infinity (clip to ±woe_clip_value)"
        ),
    )
    woe_min_bin_size: float = Field(
        0.05,
        description="Minimum fraction of total rows each WoE bin must contain in WoEEncoder.fit()",
    )

    # ── Feature engineering — HighCardinalityEncoder ──────────────────────────
    target_encoding_smoothing_factor: float = Field(
        10.0,
        description=(
            "Smoothing factor m for leave-one-out target encoding in HighCardinalityEncoder — "
            "larger values reduce overfitting on rare categories"
        ),
    )

    # ── Feature engineering — SmartImputer (missing indicator) ───────────────
    imputer_missing_indicator_threshold: float = Field(
        0.05,
        description=(
            "Missing-value fraction above which SmartImputer appends a binary {col}_was_missing "
            "indicator column"
        ),
    )

    # ── Explainability — SHAPExplainer ───────────────────────────────────────
    shap_top_n: int = Field(
        20, description="Max features shown in SHAP summary and waterfall plots"
    )
    shap_background_samples: int = Field(
        100,
        description=(
            "Max rows sampled as KernelExplainer/LinearExplainer background when no is supplied "
            "background_data"
        ),
    )
    shap_kernel_nsamples: int = Field(
        100,
        description=(
            "nsamples passed to KernelExplainer.shap_values() — higher = more accurate but slower"
        ),
    )
    shap_bootstrap_batches: int = Field(
        10, description="Number of bootstrap batches in BootstrapSHAPExplainer"
    )
    shap_bootstrap_batch_size: int = Field(
        2000, description="Rows sampled per batch in BootstrapSHAPExplainer"
    )
    shap_beeswarm_max_display_samples: int = Field(
        2000,
        description=(
            "Max data points shown per feature in BootstrapSHAPExplainer beeswarm plot; when "
            "subsampled total exceeds this"
        ),
    )

    # ── Explainability — PermutationImportanceAnalyser ──────────────────────
    permutation_importance_n_repeats: int = Field(
        10,
        description=(
            "Default n_repeats for sklearn.inspection.permutation_importance when "
            "PermutationImportanceAnalyser is used standalone (n_repeats=None)"
        ),
    )
    permutation_importance_sample_size: int = Field(
        5000,
        description=(
            "Default max rows sampled for permutation importance when "
            "PermutationImportanceAnalyser is used standalone (sample_size=None)"
        ),
    )

    # ── Explainability — PDPAnalyser ─────────────────────────────────────────
    pdp_grid_resolution: int = Field(
        50,
        description="Default number of grid points per numeric feature in PDPAnalyser.fit()",
    )
    pdp_sample_size: int = Field(
        2000,
        description="Max rows subsampled from X for marginalisation in PDPAnalyser.fit()",
    )
    pdp_min_slope: float = Field(
        0.001,
        description="Default minimum |Δprob/Δfeature| for PDPAnalyser.discover_thresholds()",
    )
    pdp_min_prob_increase: float = Field(
        0.01,
        description="Min cumulative prob gain above baseline to flag a threshold in PDPAnalyser",
    )

    # ── Feature selection ─────────────────────────────────────────────────────
    rfe_logreg_max_iter: int = Field(
        200,
        description=(
            "max_iter for the default LogisticRegression estimator used by RFESelector when no "
            "custom estimator is supplied"
        ),
    )
    shap_selector_min_importance: float = Field(
        0.0, description="Features with mean |SHAP| strictly below this are dropped by SHAPSelector"
    )
    shap_selector_max_samples: int = Field(
        2000,
        description=(
            "Max rows subsampled from X before calling shap.Explainer in SHAPSelector.fit()"
        ),
    )

    # ── Data splitting ────────────────────────────────────────────────────────
    splitter_oot_min_rows: int = Field(
        50, description="Minimum OOT rows required for temporal split — fewer rows raise ValueError"
    )
    splitter_class_ratio_tolerance: float = Field(
        0.05,
        description=(
            "Max allowed absolute difference in event rate between train and val/test splits "
            "before warning"
        ),
    )

    # ── Feature engineering — WinsorizationTransformer ───────────────────────
    winsorizer_lower_tail: float = Field(
        0.01,
        description=(
            "Lower percentile tail clipped by WinsorizationTransformer (e.g. 0.01 = 1st percentile)"
        ),
    )
    winsorizer_upper_tail: float = Field(
        0.01,
        description=(
            "Upper percentile tail clipped by WinsorizationTransformer (e.g. 0.01 = 99th "
            "percentile)"
        ),
    )

    # ── Feature engineering — StatisticalOutlierCapper ───────────────────────
    outlier_capper_iqr_multiplier: float = Field(
        1.5,
        description="IQR multiplier used by StatisticalOutlierCapper (method='iqr') to set the "
        "lower/upper cap: Q1 - k*IQR / Q3 + k*IQR",
    )
    outlier_capper_zscore_threshold: float = Field(
        3.0,
        description="Standard-deviation threshold used by StatisticalOutlierCapper "
        "(method='zscore') to set the lower/upper cap: mean ± k*std",
    )
    outlier_capper_mad_threshold: float = Field(
        3.5,
        description="Scaled-MAD threshold used by StatisticalOutlierCapper (method='mad') to "
        "set the lower/upper cap: median ± k*1.4826*MAD",
    )

    # ── Feature engineering — RareCategoryGrouper ────────────────────────────
    rare_category_min_frequency: float = Field(
        0.01,
        description="Min row-fraction a category must represent to stay as-is in "
        "RareCategoryGrouper (strategy='min_frequency') — below this, grouped into 'Other'",
    )
    rare_category_top_n: int = Field(
        10,
        description="Max categories kept per column by RareCategoryGrouper (strategy='top_n') — "
        "remainder grouped into the 'Other' bucket",
    )

    # ── Feature engineering — MultivariateImputer ────────────────────────────
    knn_imputer_n_neighbors: int = Field(
        5,
        description="Number of nearest neighbours averaged by MultivariateImputer "
        "(strategy='knn') to estimate each missing value",
    )
    iterative_imputer_max_iter: int = Field(
        10,
        description="Max round-robin regression passes performed by MultivariateImputer "
        "(strategy='iterative') before stopping",
    )

    # ── Feature engineering — DistributionTransformer (quantile strategies) ──
    quantile_transformer_n_quantiles: int = Field(
        1000,
        description="Max quantiles used by DistributionTransformer's 'quantile_uniform'/"
        "'quantile_normal' strategies; capped to the training row count when smaller",
    )

    # ── Feature engineering — PolynomialFeaturesTransformer ──────────────────
    polynomial_features_degree: int = Field(
        2,
        description="Max polynomial degree generated by PolynomialFeaturesTransformer "
        "(e.g. 2 -> x, x^2, and pairwise interaction terms)",
    )

    # ── Feature engineering — SplineFeatureTransformer ───────────────────────
    spline_transformer_n_knots: int = Field(
        5,
        description="Number of knots used by SplineFeatureTransformer's B-spline basis per column",
    )
    spline_transformer_degree: int = Field(
        3,
        description="B-spline polynomial degree used by SplineFeatureTransformer",
    )

    # ── Feature engineering — HashEncoder ─────────────────────────────────────
    hash_encoder_n_components: int = Field(
        8,
        description="Number of hashed output columns per source column in HashEncoder — fixed "
        "regardless of the column's actual cardinality",
    )

    # ── Feature engineering — NoiseInjector ───────────────────────────────────
    noise_numeric_scale: float = Field(
        0.05,
        description="Fraction of a numeric column's own standard deviation used as additive "
        "noise magnitude in NoiseInjector — train-time data augmentation only",
    )
    noise_affected_row_frac: float = Field(
        0.05,
        description="Fraction of rows perturbed per column by NoiseInjector, independently "
        "sampled per column",
    )

    # ── Feature engineering — CategoryCombinerTransformer ────────────────────
    category_combiner_cardinality_warn_threshold: int = Field(
        100,
        description="Distinct-value count above which CategoryCombinerTransformer logs a "
        "cardinality warning for its combined output column",
    )

    # ── Interactive UI — Imbalance display (Step 7) ───────────────────────────
    imbalance_balanced_min_event_rate: float = Field(
        0.35,
        description="Training-set event rate at-or-above which classes are considered "
        "roughly balanced (symmetric band, e.g. 0.35 means 35-65% is 'balanced') — drives "
        "the Interactive-mode imbalance-warning banner on Step 7",
    )

    # ── Interactive UI — Evaluate display (Step 10) ───────────────────────────
    evaluate_leakage_auc_threshold: float = Field(
        0.99,
        description="Primary-split AUC-ROC above which Step 10 flags possible data leakage "
        "(ported from interactive_step_evaluate.py's hardcoded 0.99)",
    )
    evaluate_near_random_auc_threshold: float = Field(
        0.55,
        description="Primary-split AUC-ROC below which Step 10 flags the model as close to "
        "random guessing (ported from interactive_step_evaluate.py's hardcoded 0.55)",
    )
    evaluate_overfitting_gap_threshold: float = Field(
        0.10,
        description="Train-minus-primary-split AUC gap above which Step 10 flags likely "
        "overfitting (ported from interactive_step_evaluate.py's hardcoded 0.10)",
    )
    evaluate_psi_unstable_threshold: float = Field(
        0.25,
        description="Primary-split PSI above which Step 10 flags an unstable score "
        "distribution shift (ported from interactive_step_evaluate.py's hardcoded 0.25)",
    )
    evaluate_psi_monitor_threshold: float = Field(
        0.10,
        description="Primary-split PSI above which Step 10 flags the monitoring zone, below "
        "evaluate_psi_unstable_threshold (ported from interactive_step_evaluate.py's "
        "hardcoded 0.10)",
    )

    # ── Interactive UI — Calibration display (Step 11) ────────────────────────
    calibration_ece_improvement_threshold: float = Field(
        0.001,
        description="ECE delta magnitude below which calibration's effect is considered "
        "negligible (ported from interactive_step_calibration.py's hardcoded 0.001)",
    )

    # ── Interactive UI — SHAP display (Step 12) ───────────────────────────────
    shap_interactive_max_rows: int = Field(
        500,
        description="Max rows sampled from the held-out split before running SHAP in "
        "Interactive mode / the API — keeps computation responsive (ported from "
        "interactive_step_shap.py's hardcoded _SHAP_MAX_ROWS=500)",
    )

    # ── Target — ImbalanceHandler ──────────────────────────────────────────────
    imbalance_smote_k_neighbors: int = Field(
        5,
        description="Number of nearest same-class neighbors SMOTE interpolates between "
        "when generating synthetic minority-class rows",
    )

    # ── MCP server ──────────────────────────────────────────────────────────
    mcp_output_dir: str = Field(
        "outputs/mcp",
        description="Base directory for MCP tool call run artifacts (saved models, "
        "splits, charts, reports). Each tool call gets its own uniquely-named "
        "subdirectory under this path.",
    )
    mcp_high_missing_red_flag_threshold: float = Field(
        0.5,
        description="Missing-value rate above which analyze_dataset's summary flags a "
        "column as a red flag.",
    )

    model_config = {"env_prefix": "DSCOMPANION_", "env_file": ".env", "extra": "ignore"}


# Singleton — import this everywhere
settings = DSCompanionConfig()
