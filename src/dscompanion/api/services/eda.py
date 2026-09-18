"""Step 4 (Explore the Data / EDA) service functions — the REST equivalent of
``dscompanion/app/interactive_step4.py``, with every ``streamlit`` call stripped out. Read-only:
no confirm-before-apply gate is needed since nothing here mutates the data, but Step 4 still
follows the preview/confirm shape (kept symmetric with every other step in
this API) — confirm is a lightweight flag-flip plus audit entries.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from dscompanion.api.config import api_settings
from dscompanion.api.schemas import (
    EdaColumnChartRequest,
    EdaColumnChartResponse,
    EdaConfirmRequest,
    EdaConfirmResponse,
    EdaPreviewResponse,
    EdaThresholds,
)
from dscompanion.api.state import RunState
from dscompanion.api.steps import next_step_id
from dscompanion.config import settings
from dscompanion.eda import EDAReport
from dscompanion.split import DataSplit

logger = logging.getLogger(__name__)

__all__ = ["preview_eda", "get_column_chart", "confirm_eda"]


def _exclude_identifier_columns(split: DataSplit, identifier_columns: list[str]) -> DataSplit:
    """Returns a copy of ``split`` with identifier columns dropped from every X partition.

    Direct port of ``interactive_step4.py``'s helper of the same name — identifier
    columns (e.g. a customer/account ID) chosen at Step 2 are never useful for EDA.

    Args:
        split (DataSplit): Split confirmed at Step 3.
        identifier_columns (list[str]): Columns to drop. No-op when empty.

    Returns:
        DataSplit: A new ``DataSplit`` with identifier columns absent from every feature
        partition, or ``split`` itself, unmodified, when ``identifier_columns`` is empty.
    """
    if not identifier_columns:
        return split
    return dataclasses.replace(
        split,
        train_X=split.train_X.drop(columns=identifier_columns, errors="ignore"),
        val_X=split.val_X.drop(columns=identifier_columns, errors="ignore"),
        test_X=split.test_X.drop(columns=identifier_columns, errors="ignore"),
        oot_X=split.oot_X.drop(columns=identifier_columns, errors="ignore"),
    )


def _fig_to_json(fig: go.Figure) -> dict[str, Any]:
    """JSON-safe Plotly figure dict via Plotly's own encoder (handles numpy arrays and
    datetimes correctly — more robust than ``fig.to_dict()``, same reasoning as
    ``load_data.py``'s ``_sample_rows()`` using pandas' ``to_json`` over ``to_dict``).
    """
    return json.loads(fig.to_json())


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """JSON-safe records via pandas' own JSON encoder (handles NaN/Timestamp)."""
    return json.loads(df.to_json(orient="records"))


def _get_report(run: RunState) -> EDAReport:
    """Fetches this run's cached, fitted ``EDAReport`` — set by ``preview_eda``.

    Args:
        run (RunState): The run to read from.

    Returns:
        EDAReport: The cached report.

    Raises:
        ValueError: If Step 4's preview hasn't been called yet.
    """
    report = run.artifacts.get("eda_report")
    if report is None:
        raise ValueError("Step 4 preview must be called before this endpoint.")
    return report


def preview_eda(run: RunState) -> EdaPreviewResponse:
    """Runs the full ``EDAReport`` and caches it on the run for subsequent column-chart
    and confirm calls. Always recomputes (never trusts a stale cache) — matches Steps
    1-3's preview convention of recomputing fresh every call.

    Args:
        run (RunState): The run holding Step 1's split... via Step 3's confirmed split,
            Step 2's target, and Step 2's identifier columns.

    Returns:
        EdaPreviewResponse: Overview, missing-values check, univariate summaries,
        bivariate IV table, and multivariate correlation/VIF/Cramér's V tables.

    Raises:
        ValueError: If Steps 1-3 haven't all been confirmed for this run.
    """
    split = run.artifacts.get("split")
    target = run.artifacts.get("target")
    if split is None or target is None:
        raise ValueError("Steps 1-3 must be confirmed before Step 4.")

    identifier_columns = run.artifacts.get("identifier_columns") or []
    split = _exclude_identifier_columns(split, identifier_columns)

    report = EDAReport(
        split=split,
        target=target,
        high_missing_threshold=api_settings.eda_high_missing_threshold,
        near_zero_variance_threshold=settings.near_zero_variance_threshold,
    ).run_all()
    run.artifacts["eda_report"] = report

    alerts = report.alerts()
    high_missing_alerts = [
        {"feature": feature, "detail": alert["detail"]}
        for feature, feature_alerts in alerts.items()
        for alert in feature_alerts
        if alert["type"] == "HIGH_MISSING"
    ]

    corr_fig = report.missing_correlation_heatmap()
    missing_correlation_heatmap = _fig_to_json(corr_fig) if corr_fig.data else None

    try:
        pearson_heatmap = _fig_to_json(report.pearson_heatmap())
        high_correlation_pairs = _df_to_records(report.flag_high_correlation())
        vif_table = _df_to_records(report.vif_table())
    except ValueError:
        # Fewer than 2 numeric columns — skip silently, matching interactive_step4.py.
        pearson_heatmap = None
        high_correlation_pairs = []
        vif_table = []

    logger.info("run_id=%s previewed Step 4 EDA", run.run_id)
    return EdaPreviewResponse(
        overview=report.overview_summary(),
        missing_values_chart=_fig_to_json(report.missing_values_chart()),
        missing_correlation_heatmap=missing_correlation_heatmap,
        high_missing_alerts=high_missing_alerts,
        numeric_summary=_df_to_records(report.numeric_summary()),
        categorical_summary=_df_to_records(report.categorical_summary()),
        iv_table=_df_to_records(report.iv_table()),
        pearson_heatmap=pearson_heatmap,
        high_correlation_pairs=high_correlation_pairs,
        vif_table=vif_table,
        cramers_v_table=_df_to_records(report.cramers_v_table()),
        thresholds=EdaThresholds(
            near_zero_variance=settings.near_zero_variance_threshold,
            high_cardinality=settings.high_cardinality_threshold,
            correlation=settings.correlation_threshold,
            vif=settings.vif_threshold,
        ),
    )


def get_column_chart(run: RunState, request: EdaColumnChartRequest) -> EdaColumnChartResponse:
    """Builds the per-column chart for whichever column the user picked — mirrors
    ``interactive_step4.py``'s three selectbox-driven charts, computed on demand rather
    than upfront for every column (only one is ever viewed at a time).

    Args:
        run (RunState): The run holding the cached ``EDAReport`` (via ``preview_eda``).
        request (EdaColumnChartRequest): Which chart kind and column/feature.

    Returns:
        EdaColumnChartResponse: The chart, plus a caption for ``bivariate_rate``.

    Raises:
        ValueError: If Step 4's preview hasn't been called yet, or the column doesn't exist.
    """
    report = _get_report(run)

    if request.kind == "numeric_kde":
        fig = report.numeric_distribution_with_kde(request.column)
        return EdaColumnChartResponse(figure=_fig_to_json(fig))

    if request.kind == "categorical_bar":
        counts = report.categorical_value_counts(request.column)
        fig = go.Figure(
            go.Bar(
                x=counts.index.astype(str).tolist(),
                y=counts.values.tolist(),
                marker_color="#4c9be8",
            )
        )
        fig.update_layout(
            xaxis_title=request.column,
            yaxis_title="Count",
            height=420,
            margin={"t": 30, "b": 160, "l": 100, "r": 20},
        )
        return EdaColumnChartResponse(figure=_fig_to_json(fig))

    # bivariate_rate
    fig = report.target_rate_by_bin_chart(request.column)
    iv_by_feature = report.iv_table().set_index("feature")
    caption = None
    if request.column in iv_by_feature.index:
        power = iv_by_feature.loc[request.column, "predictive_power"]
        caption = f"predictive power: {power}"
    return EdaColumnChartResponse(figure=_fig_to_json(fig), caption=caption)


def confirm_eda(run: RunState, request: EdaConfirmRequest) -> EdaConfirmResponse:
    """Marks Step 4 confirmed — a lightweight flag-flip plus audit entries, since EDA
    itself never mutates any data.

    Args:
        run (RunState): The run to persist into.
        request (EdaConfirmRequest): Whether the data quality check was skipped entirely.

    Returns:
        EdaConfirmResponse: Confirmation and the resulting step status.

    Raises:
        ValueError: If not skipped, and Step 4's preview hasn't been called yet.
    """
    if request.skipped:
        run.artifacts.pop("eda_report", None)
        status = "flagged"
        audit_entries = ["Skipped the data quality check."]
    else:
        report = run.artifacts.get("eda_report")
        if report is None:
            raise ValueError("Step 4 preview must be run before confirming (unless skipped).")
        alerts = report.alerts()
        n_high_missing = sum(
            1
            for feature_alerts in alerts.values()
            for alert in feature_alerts
            if alert["type"] == "HIGH_MISSING"
        )
        status = "done"
        audit_entries = [
            (
                f"Missing Values check: {n_high_missing} column(s) flagged."
                if n_high_missing
                else "Missing Values check: no issues found."
            ),
            "Univariate EDA check: reviewed stats and distribution charts.",
            "Bivariate EDA check: reviewed feature-vs-target IV and rate charts.",
            "Multivariate EDA check: reviewed Pearson heatmap, flagged pairs, VIF, and Cramér's V.",
        ]

    run.step_data["eda"] = {"skipped": request.skipped}
    run.step_confirmed["eda"] = True
    run.step_status["eda"] = status
    nxt = next_step_id("eda")
    if nxt is not None:
        run.step_status[nxt] = "current"
    run.audit_trail.extend(("eda", entry) for entry in audit_entries)

    logger.info("run_id=%s confirmed Step 4 EDA: status=%s", run.run_id, status)
    return EdaConfirmResponse(confirmed=True, status=status)
