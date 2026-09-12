"""HTML section renderers for every non-EDA ModelCard section — performance,
decile/lift, calibration, explainability, stability, hyperparameter tuning,
leaderboard, feature inventory, full config, governance.

Each function takes the exact ``sections_[key]`` payload ``ModelCard.generate()``
already computes for Excel/Word — no new computation here, only a new HTML
rendering path. Same visual language as ``html_eda.py``/``html_widgets.py``:
Bootstrap 5 tables, inline-SVG charts, ``section-header``/``section-items``
containers.
"""

from __future__ import annotations

import html as _html_lib
import logging
from typing import Any

logger = logging.getLogger(__name__)

import pandas as pd

from dscompanion.docs import html_widgets as hw

__all__ = [
    "render_performance_section",
    "render_decile_section",
    "render_calibration_section",
    "render_explainability_section",
    "render_stability_section",
    "render_tuning_section",
    "render_leaderboard_section",
    "render_feature_inventory_section",
    "render_full_config_section",
    "render_governance_section",
    "render_config_summary_section",
    "render_overview_section",
]


def _dataframe_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a generic Bootstrap table (header + rows), any column set."""
    if df.empty:
        return '<p class="text-muted p-3">No data.</p>'
    hdr = "".join(f"<th>{_html_lib.escape(str(c))}</th>" for c in df.columns)
    rows = "".join(
        "<tr>" + "".join(f"<td>{_html_lib.escape(str(v))}</td>" for v in row) + "</tr>"
        for row in df.itertuples(index=False)
    )
    return (
        f'<div class="table-responsive"><table class="table table-striped">'
        f"<thead><tr>{hdr}</tr></thead><tbody>{rows}</tbody></table></div>"
    )


def render_performance_section(data: pd.DataFrame) -> str:
    """Render the Performance Metrics section: one row per metric, one column per split.

    Args:
        data (pd.DataFrame): ``ModelCard.sections_["performance_metrics"]`` —
            column ``metric`` plus one column per evaluated split.

    Returns:
        str: HTML table, or a "no metrics" note when ``data`` is empty.
    """
    if data.empty:
        return '<p class="text-muted p-3">No performance metrics available for this run.</p>'
    return _dataframe_table(data)


def render_decile_section(data: pd.DataFrame) -> str:
    """Render the Decile / Lift section: gains table + event-rate/cumulative-lift combo chart.

    Args:
        data (pd.DataFrame): ``ModelCard.sections_["decile_table"]`` — see
            ``dscompanion.utils.metrics.DECILE_TABLE_COLUMNS``.

    Returns:
        str: HTML table + SVG chart, or a "no decile table" note when
        ``data`` is empty.
    """
    if data.empty:
        return '<p class="text-muted p-3">No decile table available for this run.</p>'
    chart = hw.combo_bar_line_svg(
        categories=[str(d) for d in data["decile"]],
        bar_values=data["event_rate"].tolist(),
        bar_label="event_rate",
        line_values=data["cumulative_lift"].tolist(),
        line_label="cumulative_lift",
        title="Decile Table — Event Rate & Cumulative Lift",
    )
    return f'<div class="mb-3">{chart}</div>{_dataframe_table(data)}'


def render_calibration_section(data: dict[str, Any]) -> str:
    """Render the Calibration section: method + Expected Calibration Error before/after.

    Args:
        data (dict): ``ModelCard.sections_["calibration"]`` — either
            ``{"method": str, "ece_before": float, "ece_after": float}`` or
            ``{"note": str}`` when no calibrator was provided.

    Returns:
        str: HTML stat table, or the note text when calibration is
        unavailable.
    """
    if "note" in data:
        return f'<p class="text-muted p-3">{_html_lib.escape(data["note"])}</p>'
    rows = (
        hw.stat_row("Method", data.get("method"))
        + hw.stat_row("ECE before calibration", data.get("ece_before"))
        + hw.stat_row("ECE after calibration", data.get("ece_after"))
    )
    return hw.table(rows)


def render_explainability_section(data: dict[str, Any]) -> str:
    """Render the Explainability section: SHAP and/or permutation importance bar charts.

    Args:
        data (dict): ``ModelCard.sections_["explainability"]`` — any
            combination of ``{"top_features": list[dict]}`` (SHAP, each dict
            with ``feature``, ``mean_abs_shap``, ``rank``) and
            ``{"permutation_top_features": list[dict]}`` (each dict with
            ``feature``, ``importance_mean``, ``importance_std``, ``rank``),
            or ``{"note": str}`` when neither is available.

    Returns:
        str: HTML bar chart(s) + table(s) for whichever of SHAP/permutation
        importance are present, or the note text when neither is available.
    """
    blocks: list[str] = []
    if data.get("top_features"):
        top = data["top_features"]
        chart = hw.horizontal_bar_svg(
            labels=[r["feature"] for r in top],
            values=[r["mean_abs_shap"] for r in top],
            title="Mean |SHAP| Feature Importance",
        )
        blocks.append(f'<div class="mb-3">{chart}</div>{_dataframe_table(pd.DataFrame(top))}')
    if data.get("permutation_top_features"):
        perm = data["permutation_top_features"]
        chart = hw.horizontal_bar_svg(
            labels=[r["feature"] for r in perm],
            values=[r["importance_mean"] for r in perm],
            title="Permutation Feature Importance",
        )
        blocks.append(f'<div class="mb-3">{chart}</div>{_dataframe_table(pd.DataFrame(perm))}')
    if not blocks:
        note = data.get("note", "No SHAP explainer or permutation importance provided.")
        return f'<p class="text-muted p-3">{_html_lib.escape(note)}</p>'
    return "".join(blocks)


def render_stability_section(data: dict[str, Any]) -> str:
    """Render the Stability section: per-feature PSI bar chart + flagged-feature count.

    Args:
        data (dict): ``ModelCard.sections_["stability"]`` — either
            ``{"feature_psi": list[dict], "flagged_count": int,
            "total_features": int}`` or ``{"note": str}``.

    Returns:
        str: HTML bar chart + table, or the note text when unavailable.
    """
    if "note" in data:
        return f'<p class="text-muted p-3">{_html_lib.escape(data["note"])}</p>'
    psi_rows = data["feature_psi"]
    summary = hw.stat_row("Flagged features", f'{data["flagged_count"]} / {data["total_features"]}')
    chart = hw.horizontal_bar_svg(
        labels=[r["feature"] for r in psi_rows],
        values=[r["psi"] for r in psi_rows],
        title="Population Stability Index by Feature",
    )
    table_html = _dataframe_table(pd.DataFrame(psi_rows))
    return f"{hw.table(summary)}<div class='mt-3 mb-3'>{chart}</div>{table_html}"


def render_tuning_section(data: dict[str, Any]) -> str:
    """Render the Hyperparameter Tuning section: backend, trial count, best params/score.

    Args:
        data (dict): ``ModelCard.sections_["hyperparameter_tuning"]`` —
            ``{"backend": str, "n_trials": int, "metric": str, "best_params":
            dict, "best_score": float}`` (any subset, best-effort) or
            ``{"note": str}``.

    Returns:
        str: HTML stat table(s), or the note text when unavailable.
    """
    if "note" in data:
        return f'<p class="text-muted p-3">{_html_lib.escape(data["note"])}</p>'
    best_params = data.get("best_params", {}) or {}
    top_rows = (
        hw.stat_row("Backend", data.get("backend"))
        + hw.stat_row("Trials", data.get("n_trials"))
        + hw.stat_row("Metric", data.get("metric"))
        + hw.stat_row("Best score", data.get("best_score"))
    )
    params_rows = "".join(hw.stat_row(_html_lib.escape(str(k)), v) for k, v in best_params.items())
    parts = [hw.table(top_rows)]
    if params_rows:
        parts.append(f'<p class="h5 item-header mt-3">Best parameters</p>{hw.table(params_rows)}')
    return "".join(parts)


def render_leaderboard_section(data: pd.DataFrame) -> str:
    """Render the Leaderboard section: ranked multi-algorithm comparison table.

    Args:
        data (pd.DataFrame): ``ModelCard.sections_["leaderboard"]`` — columns
            vary by run (``algorithm``, ``status``, ``fit_time_seconds``,
            ``error``, plus the configured ranking metric); rendered
            generically regardless of exact column set.

    Returns:
        str: HTML table, or a "no leaderboard" note when ``data`` is empty.
    """
    if data.empty:
        return '<p class="text-muted p-3">Leaderboard mode was not enabled for this run.</p>'
    return _dataframe_table(data)


def render_feature_inventory_section(data: pd.DataFrame) -> str:
    """Render the Feature Inventory section: every feature considered for modelling.

    Args:
        data (pd.DataFrame): ``ModelCard.sections_["feature_inventory"]`` —
            columns ``feature``, ``dtype``.

    Returns:
        str: HTML table.
    """
    return _dataframe_table(data)


def render_full_config_section(data: dict[str, Any]) -> str:
    """Render the Full Config section: the verbatim experiment YAML.

    Args:
        data (dict): ``ModelCard.sections_["full_config"]`` — either
            ``{"yaml": str}`` or ``{"note": str}``.

    Returns:
        str: A ``<pre>`` block with the YAML, or the note text.
    """
    if "yaml" not in data:
        return f'<p class="text-muted p-3">{_html_lib.escape(data.get("note", ""))}</p>'
    yaml_text = _html_lib.escape(data["yaml"])
    return (
        f'<pre class="small p-3" style="background:#f8f9fa;border-radius:6px">' f"{yaml_text}</pre>"
    )


def render_governance_section(governance_df: pd.DataFrame, limitations: dict[str, Any]) -> str:
    """Render the Governance & Limitations section: sign-off table + boilerplate notes.

    Args:
        governance_df (pd.DataFrame): ``ModelCard.sections_["governance"]`` —
            columns ``role``, ``name``, ``date``, ``sign_off``.
        limitations (dict): ``ModelCard.sections_["limitations"]`` —
            ``{"notes": str, "boilerplate": str}``.

    Returns:
        str: HTML sign-off table + limitations text.
    """
    table_html = _dataframe_table(governance_df)
    notes = _html_lib.escape(limitations.get("notes", "") or "—")
    boilerplate = _html_lib.escape(limitations.get("boilerplate", ""))
    return f"""
    {table_html}
    <p class="h5 item-header mt-3">Limitations</p>
    <p>{notes}</p>
    <p class="text-body-secondary small">{boilerplate}</p>"""


def render_config_summary_section(data: pd.DataFrame) -> str:
    """Render the Config Summary section: every monitored parameter, deviations flagged.

    Args:
        data (pd.DataFrame): ``ModelCard.sections_["config_summary"]`` —
            columns ``parameter``, ``choices``, ``default``, ``user_choice``,
            ``is_deviation``.

    Returns:
        str: HTML table, or a "no config checks" note when ``data`` is
        empty.
    """
    if data.empty:
        return (
            '<p class="text-muted p-3">No config deviation checks were captured for this run.</p>'
        )
    return _dataframe_table(data)


def render_overview_section(
    model_summary: dict[str, Any], data_lineage: dict[str, Any], config_summary: pd.DataFrame
) -> str:
    """Render the report's top-level Overview section: model summary + data lineage + deviations.

    Args:
        model_summary (dict): ``ModelCard.sections_["model_summary"]`` — a
            flat ``parameter: value`` dict, plus one nested-dict value,
            ``package_versions`` (R.3), rendered as its own sub-table below
            the flat one rather than through the generic ``stat_row(v)``
            loop, which would otherwise dump its ``repr()`` into one cell —
            same pattern ``render_tuning_section()`` uses for
            ``best_params``.
        data_lineage (dict): ``ModelCard.sections_["data_lineage"]``.
        config_summary (pd.DataFrame): ``ModelCard.sections_["config_summary"]``,
            used only to compute the deviation count shown here (the full
            table is a separate navbar section).

    Returns:
        str: HTML for a two-column Bootstrap row.
    """
    package_versions = model_summary.get("package_versions", {}) or {}
    model_rows = "".join(
        hw.stat_row(_html_lib.escape(k.replace("_", " ").title()), v)
        for k, v in model_summary.items()
        if k != "package_versions"
    )
    package_rows = "".join(hw.stat_row(_html_lib.escape(k), v) for k, v in package_versions.items())
    lineage_rows = "".join(
        hw.stat_row(_html_lib.escape(k.replace("_", " ").title()), v)
        for k, v in data_lineage.items()
    )
    n_deviations = int(config_summary["is_deviation"].sum()) if not config_summary.empty else 0
    deviation_banner = (
        f'<div class="alert alert-warning py-2 small mb-3">{n_deviations} non-default config '
        f"choice{'s' if n_deviations != 1 else ''} — see Config Summary.</div>"
        if n_deviations > 0
        else '<div class="alert alert-success py-2 small mb-3">No non-default config choices.</div>'
    )
    package_versions_html = (
        f'<p class="h5 item-header mt-3">Package versions</p>{hw.table(package_rows)}'
        if package_rows
        else ""
    )
    return f"""
    {deviation_banner}
    <div class="row sub-item mt-2">
      <div class="col-sm-6">
        <p class="h4 item-header">Model summary</p>
        {hw.table(model_rows)}
        {package_versions_html}
      </div>
      <div class="col-sm-6">
        <p class="h4 item-header">Data lineage</p>
        {hw.table(lineage_rows)}
      </div>
    </div>"""
