"""Utility helpers for metrics, validation, plotting, and synthetic data."""

from dscompanion.utils.metrics import (
    expected_calibration_error,
    feature_psi_table,
    gini_coefficient,
    iv_score,
    ks_statistic,
    psi_score,
    woe_bins,
)
from dscompanion.utils.plotting import apply_dscompanion_theme, fig_to_base64
from dscompanion.utils.synthetic import SyntheticDataGenerator
from dscompanion.utils.validators import (
    is_databricks,
    validate_binary_target,
    validate_dataframe,
)

__all__ = [
    "ks_statistic",
    "gini_coefficient",
    "psi_score",
    "feature_psi_table",
    "iv_score",
    "woe_bins",
    "expected_calibration_error",
    "validate_dataframe",
    "validate_binary_target",
    "is_databricks",
    "apply_dscompanion_theme",
    "fig_to_base64",
    "SyntheticDataGenerator",
]
