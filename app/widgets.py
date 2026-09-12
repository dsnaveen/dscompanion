"""Config-editing widgets: curated core fields + an Advanced expander for the rest.

Local-dev Streamlit UI only — never imported by dscompanion's runtime code. Widget
bounds/defaults mirror the pydantic validators in ``dscompanion.pipeline.config``
exactly, via the small helper functions below, so the UI can never drift from
what ``PipelineConfig`` actually accepts.
"""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st
import yaml

from dscompanion.models import ModelFactory

logger = logging.getLogger(__name__)

__all__ = ["render_core_widgets", "render_advanced_widgets", "build_config_dict"]


def _frac_open01(label: str, key: str, default: float) -> None:
    """Render a slider for a float validated to the open interval (0, 1).

    Args:
        label (str): Widget label.
        key (str): Flat ``cfg.*`` session-state key.
        default (float): Default value if unset.

    Returns:
        None
    """
    st.slider(label, min_value=0.01, max_value=0.99, value=default, step=0.01, key=key)


def _clip_frac(label: str, key: str, default: float) -> None:
    """Render a slider for a float validated to [0.0, 0.5).

    Args:
        label (str): Widget label.
        key (str): Flat ``cfg.*`` session-state key.
        default (float): Default value if unset.

    Returns:
        None
    """
    st.slider(label, min_value=0.0, max_value=0.49, value=default, step=0.01, key=key)


def _nonneg_float(label: str, key: str, default: float) -> None:
    """Render a number input for a float validated to be >= 0.

    Args:
        label (str): Widget label.
        key (str): Flat ``cfg.*`` session-state key.
        default (float): Default value if unset.

    Returns:
        None
    """
    st.number_input(label, min_value=0.0, value=default, key=key)


def _positive_int(label: str, key: str, default: int) -> None:
    """Render a number input for an int validated to be >= 1.

    Args:
        label (str): Widget label.
        key (str): Flat ``cfg.*`` session-state key.
        default (int): Default value if unset.

    Returns:
        None
    """
    st.number_input(label, min_value=1, value=default, step=1, key=key)


def render_core_widgets(columns: list[str]) -> None:
    """Render the ~15 always-visible config widgets.

    Args:
        columns (list[str]): Column names of the currently loaded dataset,
            used to populate the target-column selectbox. Empty list falls
            back to a free-text input.

    Returns:
        None
    """
    st.text_input("Experiment name", value="streamlit_run", key="cfg.name")

    if columns:
        st.selectbox("Target column", columns, key="cfg.data.target")
    else:
        st.text_input("Target column", key="cfg.data.target")

    task = st.selectbox(
        "Task", ["classification", "regression", "clustering"], key="cfg.model.task"
    )
    algo_options = ModelFactory.SUPPORTED_ALGORITHMS[task]
    if st.session_state.get("cfg.model.algorithm") not in algo_options:
        st.session_state["cfg.model.algorithm"] = algo_options[0]
    st.selectbox("Algorithm", algo_options, key="cfg.model.algorithm")

    st.selectbox(
        "Split method",
        ["stratified", "random", "temporal", "grouped"],
        key="cfg.split.method",
    )
    _frac_open01("Test Size", "cfg.split.test_size", 0.2)
    _frac_open01("Validation Size (of Train)", "cfg.split.val_size", 0.1)

    st.selectbox(
        "Imbalance strategy",
        ["class_weight", "smote", "undersample", "oversample", "none"],
        key="cfg.target.imbalance.strategy",
    )

    _nonneg_float("IV Threshold (Selection)", "cfg.selection.iv_threshold", 0.02)
    _frac_open01("Correlation Threshold (Selection)", "cfg.selection.correlation_threshold", 0.85)

    st.toggle("Enable tuning (Optuna)", value=False, key="cfg.tuning.enabled")
    if st.session_state.get("cfg.tuning.enabled"):
        _positive_int("Tuning Trials", "cfg.tuning.n_trials", 50)

    st.toggle("Enable leaderboard (compare algorithms)", value=False, key="cfg.leaderboard.enabled")
    if st.session_state.get("cfg.leaderboard.enabled"):
        if task != "classification":
            st.warning("Leaderboard mode currently requires task = classification.")
        mode = st.radio(
            "Restrict leaderboard by",
            ["All Algorithms", "Include List", "Exclude List"],
            key="cfg.leaderboard.mode",
            horizontal=True,
        )
        if mode == "Include List":
            st.multiselect(
                "Algorithms to include",
                ModelFactory.SUPPORTED_ALGORITHMS["classification"],
                key="cfg.leaderboard.include",
            )
        elif mode == "Exclude List":
            st.multiselect(
                "Algorithms to exclude",
                ModelFactory.SUPPORTED_ALGORITHMS["classification"],
                key="cfg.leaderboard.exclude",
            )


def render_advanced_widgets(columns: list[str]) -> None:
    """Render the remaining ~80 config fields inside collapsed expanders.

    Args:
        columns (list[str]): Column names of the currently loaded dataset,
            used for the optional ``feature_columns``/``date_column``
            multiselects.

    Returns:
        None
    """
    with st.expander("Experiment identity"):
        st.text_input("Version", value="1.0", key="cfg.version")
        st.text_area("Description", key="cfg.description")
        st.text_input("Owner", key="cfg.owner")

    with st.expander("Data"):
        st.multiselect(
            "Feature columns (empty = all except target)", columns, key="cfg.data.feature_columns"
        )
        st.selectbox(
            "Date column (temporal split only)",
            [None, *columns],
            key="cfg.data.date_column",
        )
        st.number_input(
            "Row limit (dev only, blank = no limit)",
            min_value=0,
            value=0,
            key="cfg.data.nrows",
            help="0 means no limit. Not forwarded to the config.",
        )

    with st.expander("Split (advanced)"):
        st.text_input("Group column (grouped split only)", key="cfg.split.group_column")

    with st.expander("EDA"):
        st.toggle("EDA enabled", value=True, key="cfg.eda.enabled")
        st.toggle("Univariate", value=True, key="cfg.eda.univariate")
        st.toggle("Bivariate", value=True, key="cfg.eda.bivariate")
        st.toggle("Multivariate (expensive on 500+ cols)", value=False, key="cfg.eda.multivariate")
        _nonneg_float("Skewness Alert Threshold", "cfg.eda.skewness_alert_threshold", 1.0)
        _frac_open01("Zero-% Alert Threshold", "cfg.eda.zero_pct_alert_threshold", 0.05)
        _frac_open01("Imbalance Alert Threshold", "cfg.eda.imbalance_alert_threshold", 0.5)
        _positive_int("Extreme Values Shown per Feature", "cfg.eda.extreme_values_n", 5)
        _positive_int("Sample Rows Shown", "cfg.eda.sample_n_rows", 10)
        _positive_int("Duplicate Rows Max Display", "cfg.eda.duplicate_rows_max_display", 50)
        _positive_int("Missing Matrix Max Rows", "cfg.eda.missing_matrix_max_rows", 500)
        _positive_int("Interaction Max Numeric Cols", "cfg.eda.interaction_max_numeric_cols", 8)
        _positive_int(
            "Interaction Hexbin Row Threshold", "cfg.eda.interaction_hexbin_row_threshold", 2000
        )
        _positive_int("Text Analysis Top-N Words", "cfg.eda.text_analysis_top_n_words", 20)
        _positive_int("Text Analysis Top-N Chars", "cfg.eda.text_analysis_top_n_chars", 20)
        st.toggle("Chart generation enabled", value=True, key="cfg.eda.chart")
        _clip_frac("Chart Clip Lower %", "cfg.eda.chart_clip_lower_pct", 0.01)
        _clip_frac("Chart Clip Upper %", "cfg.eda.chart_clip_upper_pct", 0.01)
        st.caption("Compliance opt-ins: expose row-level data when enabled")
        st.toggle("Include sample rows", value=False, key="cfg.eda.include_sample_rows")
        st.toggle(
            "Include duplicate row content",
            value=False,
            key="cfg.eda.include_duplicate_row_content",
        )
        st.toggle(
            "Include text sample values", value=False, key="cfg.eda.include_text_sample_values"
        )

    with st.expander("Features"):
        st.selectbox(
            "Numeric imputer strategy",
            ["auto", "mean", "median", "constant"],
            key="cfg.features.imputer.numeric_strategy",
        )
        st.selectbox(
            "Categorical imputer strategy",
            ["most_frequent", "constant"],
            key="cfg.features.imputer.categorical_strategy",
        )
        st.text_input(
            "Imputer constant fill value (if strategy=constant)",
            key="cfg.features.imputer.fill_value",
        )
        st.toggle(
            "Add missing-indicator columns",
            value=False,
            key="cfg.features.imputer.add_missing_indicator",
        )
        st.selectbox(
            "Encoder strategy",
            ["ordinal", "onehot", "target"],
            key="cfg.features.encoder.strategy",
        )
        _positive_int(
            "Max Categories Before Drop (Cardinality)", "cfg.features.encoder.max_categories", 50
        )
        st.selectbox(
            "Scaler strategy",
            ["none", "standard", "minmax", "robust"],
            key="cfg.features.scaler.strategy",
        )
        st.toggle("Winsorizer enabled", value=False, key="cfg.features.winsorizer.enabled")
        _clip_frac("Winsorizer Lower Tail", "cfg.features.winsorizer.lower_tail", 0.01)
        _clip_frac("Winsorizer Upper Tail", "cfg.features.winsorizer.upper_tail", 0.01)

    with st.expander("Target (advanced)"):
        st.text_input(
            "Sampling strategy (ratio or 'auto')",
            value="auto",
            key="cfg.target.imbalance.sampling_strategy",
        )
        st.number_input(
            "Binarize threshold (blank = unused)",
            value=0.0,
            key="cfg.target.binarize_threshold",
            help="0 is treated as unset. Set a non-zero cutoff to binarize a continuous target.",
        )

    with st.expander("Selection (remaining fields)"):
        st.toggle("Remove high-null columns", value=True, key="cfg.selection.remove_high_null")
        _frac_open01("Null Rate Threshold", "cfg.selection.null_rate_threshold", 0.8)
        st.toggle("Remove constant columns", value=True, key="cfg.selection.remove_constant")
        st.toggle(
            "Remove quasi-constant columns", value=True, key="cfg.selection.remove_quasi_constant"
        )
        _frac_open01("Quasi-Constant Threshold", "cfg.selection.quasi_constant_threshold", 0.99)
        st.toggle(
            "Remove high-cardinality columns",
            value=True,
            key="cfg.selection.remove_high_cardinality",
        )
        _positive_int("Cardinality Threshold", "cfg.selection.cardinality_threshold", 50)
        st.toggle("Leakage check", value=True, key="cfg.selection.leakage_check")
        _frac_open01("Leakage Threshold", "cfg.selection.leakage_threshold", 0.95)
        st.toggle(
            "Remove high-correlation columns",
            value=True,
            key="cfg.selection.remove_high_correlation",
        )
        st.toggle("VIF enabled (expensive)", value=False, key="cfg.selection.vif_enabled")
        _nonneg_float("VIF Threshold", "cfg.selection.vif_threshold", 10.0)

    with st.expander("Model params (advanced overrides)"):
        st.text_area(
            "Inline YAML/JSON params merged on top of dscompanion's defaults, e.g. {max_depth: 4}",
            key="cfg.model.params_text",
        )

    with st.expander("Tuning (remaining fields)"):
        st.text_input("Tuning metric", value="roc_auc", key="cfg.tuning.metric")
        st.selectbox("Tuning direction", ["maximize", "minimize"], key="cfg.tuning.direction")

    with st.expander("Leaderboard (remaining fields)"):
        st.text_input("Sort metric (blank = auto)", key="cfg.leaderboard.sort_metric")
        st.selectbox(
            "Eval split (blank = auto val-else-test)",
            [None, "train", "val", "test"],
            key="cfg.leaderboard.eval_split",
        )

    with st.expander("Explainability"):
        st.toggle(
            "SHAP enabled",
            value=False,
            key="cfg.explain.shap_enabled",
            help="Off by default: SHAP computation is relatively expensive, so it's "
            "opt-in rather than run automatically.",
        )
        _positive_int("SHAP Sample Size", "cfg.explain.shap_sample_size", 5000)
        _positive_int("SHAP Top-N Features", "cfg.explain.shap_top_n", 20)
        st.toggle("LIME enabled (slower)", value=False, key="cfg.explain.lime_enabled")

    with st.expander("Reporting"):
        st.text_input("Report output directory", value="./reports", key="cfg.reporting.output_dir")
        st.toggle("Generate HTML report", value=False, key="cfg.reporting.html_report")
        st.toggle("Include decile table", value=True, key="cfg.reporting.decile_table")


def _parse_model_params() -> dict[str, Any]:
    """Parse the free-form model-params text area into a dict.

    Args:
        None

    Returns:
        dict[str, Any]: Parsed mapping, or an empty dict if the field is
        blank or fails to parse (a parse failure is surfaced separately by
        the caller via ``st.error``, this function never raises).
    """
    text = st.session_state.get("cfg.model.params_text", "")
    if not text or not text.strip():
        return {}
    try:
        parsed = yaml.safe_load(text)
        return parsed if isinstance(parsed, dict) else {}
    except yaml.YAMLError as exc:
        st.error(f"Model params could not be parsed as YAML/JSON: {exc}")
        return {}


def build_config_dict(data_path: str, data_format: str) -> dict[str, Any]:
    """Assemble the nested config dict from all ``cfg.*`` session-state keys.

    Args:
        data_path (str): Resolved path to the selected/uploaded data file.
        data_format (str): ``"csv"`` or ``"parquet"``, inferred from the
            file suffix.

    Returns:
        dict[str, Any]: Nested dict matching ``PipelineConfig``'s shape,
        ready to pass as ``PipelineConfig(**build_config_dict(...))``.
    """
    ss = st.session_state
    feature_columns = ss.get("cfg.data.feature_columns") or None
    nrows = ss.get("cfg.data.nrows") or None
    fill_value = ss.get("cfg.features.imputer.fill_value") or None
    binarize_threshold = ss.get("cfg.target.binarize_threshold") or None
    sort_metric = ss.get("cfg.leaderboard.sort_metric") or None
    eval_split = ss.get("cfg.leaderboard.eval_split") or None
    date_column = ss.get("cfg.data.date_column") or None

    return {
        "name": ss.get("cfg.name", "streamlit_run"),
        "version": ss.get("cfg.version", "1.0"),
        "description": ss.get("cfg.description", ""),
        "owner": ss.get("cfg.owner", ""),
        "data": {
            "path": data_path,
            "format": data_format,
            "target": ss.get("cfg.data.target", ""),
            "feature_columns": feature_columns,
            "date_column": date_column,
            "nrows": nrows,
        },
        "split": {
            "method": ss.get("cfg.split.method", "stratified"),
            "test_size": ss.get("cfg.split.test_size", 0.2),
            "val_size": ss.get("cfg.split.val_size", 0.1),
            "group_column": ss.get("cfg.split.group_column") or None,
        },
        "eda": {
            "enabled": ss.get("cfg.eda.enabled", True),
            "univariate": ss.get("cfg.eda.univariate", True),
            "bivariate": ss.get("cfg.eda.bivariate", True),
            "multivariate": ss.get("cfg.eda.multivariate", False),
            "skewness_alert_threshold": ss.get("cfg.eda.skewness_alert_threshold", 1.0),
            "zero_pct_alert_threshold": ss.get("cfg.eda.zero_pct_alert_threshold", 0.05),
            "imbalance_alert_threshold": ss.get("cfg.eda.imbalance_alert_threshold", 0.5),
            "extreme_values_n": ss.get("cfg.eda.extreme_values_n", 5),
            "sample_n_rows": ss.get("cfg.eda.sample_n_rows", 10),
            "duplicate_rows_max_display": ss.get("cfg.eda.duplicate_rows_max_display", 50),
            "missing_matrix_max_rows": ss.get("cfg.eda.missing_matrix_max_rows", 500),
            "interaction_max_numeric_cols": ss.get("cfg.eda.interaction_max_numeric_cols", 8),
            "interaction_hexbin_row_threshold": ss.get(
                "cfg.eda.interaction_hexbin_row_threshold", 2000
            ),
            "text_analysis_top_n_words": ss.get("cfg.eda.text_analysis_top_n_words", 20),
            "text_analysis_top_n_chars": ss.get("cfg.eda.text_analysis_top_n_chars", 20),
            "include_sample_rows": ss.get("cfg.eda.include_sample_rows", False),
            "include_duplicate_row_content": ss.get("cfg.eda.include_duplicate_row_content", False),
            "include_text_sample_values": ss.get("cfg.eda.include_text_sample_values", False),
            "chart": ss.get("cfg.eda.chart", True),
            "chart_clip_lower_pct": ss.get("cfg.eda.chart_clip_lower_pct", 0.01),
            "chart_clip_upper_pct": ss.get("cfg.eda.chart_clip_upper_pct", 0.01),
        },
        "features": {
            "imputer": {
                "numeric_strategy": ss.get("cfg.features.imputer.numeric_strategy", "auto"),
                "categorical_strategy": ss.get(
                    "cfg.features.imputer.categorical_strategy", "most_frequent"
                ),
                "fill_value": fill_value,
                "add_missing_indicator": ss.get(
                    "cfg.features.imputer.add_missing_indicator", False
                ),
            },
            "encoder": {
                "strategy": ss.get("cfg.features.encoder.strategy", "ordinal"),
                "max_categories": ss.get("cfg.features.encoder.max_categories", 50),
            },
            "scaler": {"strategy": ss.get("cfg.features.scaler.strategy", "none")},
            "winsorizer": {
                "enabled": ss.get("cfg.features.winsorizer.enabled", False),
                "lower_tail": ss.get("cfg.features.winsorizer.lower_tail", 0.01),
                "upper_tail": ss.get("cfg.features.winsorizer.upper_tail", 0.01),
            },
        },
        "target": {
            "imbalance": {
                "strategy": ss.get("cfg.target.imbalance.strategy", "class_weight"),
                "sampling_strategy": ss.get("cfg.target.imbalance.sampling_strategy", "auto"),
            },
            "binarize_threshold": binarize_threshold,
        },
        "selection": {
            "remove_high_null": ss.get("cfg.selection.remove_high_null", True),
            "null_rate_threshold": ss.get("cfg.selection.null_rate_threshold", 0.8),
            "remove_constant": ss.get("cfg.selection.remove_constant", True),
            "remove_quasi_constant": ss.get("cfg.selection.remove_quasi_constant", True),
            "quasi_constant_threshold": ss.get("cfg.selection.quasi_constant_threshold", 0.99),
            "remove_high_cardinality": ss.get("cfg.selection.remove_high_cardinality", True),
            "cardinality_threshold": ss.get("cfg.selection.cardinality_threshold", 50),
            "leakage_check": ss.get("cfg.selection.leakage_check", True),
            "leakage_threshold": ss.get("cfg.selection.leakage_threshold", 0.95),
            "remove_high_correlation": ss.get("cfg.selection.remove_high_correlation", True),
            "correlation_threshold": ss.get("cfg.selection.correlation_threshold", 0.85),
            "iv_threshold": ss.get("cfg.selection.iv_threshold", 0.02),
            "vif_enabled": ss.get("cfg.selection.vif_enabled", False),
            "vif_threshold": ss.get("cfg.selection.vif_threshold", 10.0),
        },
        "model": {
            "task": ss.get("cfg.model.task", "classification"),
            "algorithm": ss.get("cfg.model.algorithm", "xgboost"),
            "params": _parse_model_params(),
        },
        "tuning": {
            "enabled": ss.get("cfg.tuning.enabled", False),
            "n_trials": ss.get("cfg.tuning.n_trials", 50),
            "metric": ss.get("cfg.tuning.metric", "roc_auc"),
            "direction": ss.get("cfg.tuning.direction", "maximize"),
        },
        "leaderboard": {
            "enabled": ss.get("cfg.leaderboard.enabled", False),
            "include": ss.get("cfg.leaderboard.include") or None,
            "exclude": ss.get("cfg.leaderboard.exclude") or None,
            "sort_metric": sort_metric,
            "eval_split": eval_split,
        },
        "explain": {
            "shap_enabled": ss.get("cfg.explain.shap_enabled", False),
            "shap_sample_size": ss.get("cfg.explain.shap_sample_size", 5000),
            "shap_top_n": ss.get("cfg.explain.shap_top_n", 20),
            "lime_enabled": ss.get("cfg.explain.lime_enabled", False),
        },
        "reporting": {
            "output_dir": ss.get("cfg.reporting.output_dir", "./reports"),
            "html_report": ss.get("cfg.reporting.html_report", False),
            "decile_table": ss.get("cfg.reporting.decile_table", True),
        },
    }
