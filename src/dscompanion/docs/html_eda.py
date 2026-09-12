"""HTML section renderers for ModelCard's EDA report — Overview, Alerts,
Schema, per-column Variable cards, Correlations, Missing values, Sample.

Reads directly from a fitted ``EDAReport`` (never from ``sections_["eda_summary"]``
— that intermediate key was removed along with the old Jinja2/Plotly
implementation). Mirrors ``ModelCard._write_eda_excel_sheets()``'s existing
data-source pattern: same accessor methods, same "sections are skipped when
empty, never written empty" convention, different output format.
"""

from __future__ import annotations

import html as _html_lib
import logging
from typing import Any

logger = logging.getLogger(__name__)

import pandas as pd

from dscompanion.docs import html_widgets as hw

__all__ = [
    "render_overview_tab",
    "render_alerts_tab",
    "render_schema_tab",
    "render_variable_cards",
    "render_correlations_section",
    "render_missing_section",
    "render_sample_section",
]


def render_overview_tab(eda_report: Any) -> str:
    """Render the Overview tab: dataset statistics + variable-type counts.

    Args:
        eda_report (EDAReport): Fitted ``EDAReport`` instance (post
            ``run_all()``).

    Returns:
        str: HTML for a two-column Bootstrap row (stats table, type-count
        table).
    """
    ov = eda_report.overview_summary()
    dup_cnt = ov.get("duplicate_rows")
    dup_pct = ov.get("duplicate_pct")
    dup_str = f"{dup_cnt:,} ({dup_pct}%)" if isinstance(dup_cnt, int) else "—"

    stats_rows = (
        hw.stat_row("Number of variables", ov["total_columns"])
        + hw.stat_row("Number of observations", f'{ov["total_rows"]:,}')
        + hw.stat_row("Missing cells", f'{ov["total_missing"]:,}')
        + hw.stat_row("Missing cells (%)", f'{ov["missing_pct"]}%')
        + hw.stat_row("Duplicate rows", dup_str)
    )

    numeric_df = eda_report.numeric_summary()
    categorical_df = eda_report.categorical_summary()
    datetime_df = eda_report.datetime_summary()
    boolean_df = eda_report.boolean_summary()
    type_rows = "".join(
        hw.stat_row(label, len(df))
        for label, df in (
            ("Numeric", numeric_df),
            ("Categorical", categorical_df),
            ("Datetime", datetime_df),
            ("Boolean", boolean_df),
        )
        if len(df) > 0
    )

    return f"""
    <div class="row sub-item mt-2">
      <div class="col-sm-6">
        <p class="h4 item-header">Dataset statistics</p>
        {hw.table(stats_rows)}
      </div>
      <div class="col-sm-6">
        <p class="h4 item-header">Variable types</p>
        {hw.table(type_rows)}
      </div>
    </div>"""


def render_alerts_tab(eda_report: Any, feature_anchor: dict[str, str]) -> tuple[str, int]:
    """Render the Alerts tab: every flagged feature with a link to its variable card.

    Args:
        eda_report (EDAReport): Fitted ``EDAReport`` instance.
        feature_anchor (dict[str, str]): Mapping of feature name to the
            anchor id of its variable card (e.g. ``{"income": "var-3"}``),
            from ``render_variable_cards()`` (Task 3). Features missing from
            this mapping render without a link.

    Returns:
        tuple[str, int]: HTML for the alerts table, and the total alert
        count across all features.
    """
    alerts = eda_report.alerts()
    total = sum(len(v) for v in alerts.values())
    rows = []
    for feature, feature_alerts in alerts.items():
        anchor = feature_anchor.get(feature)
        name_html = (
            f'<a href="#{_html_lib.escape(anchor)}"><code>{_html_lib.escape(feature)}</code></a>'
            if anchor
            else f"<code>{_html_lib.escape(feature)}</code>"
        )
        for a in feature_alerts:
            color = hw.ALERT_BADGE_COLOR.get(a["type"], hw.DEFAULT_ALERT_COLOR)
            rows.append(
                f"<tr><td>{name_html} {_html_lib.escape(a['detail'])}</td>"
                f'<td><span class="badge" style="background:{color}">'
                f"{_html_lib.escape(a['type'])}</span></td></tr>"
            )
    rows_html = "".join(rows) or (
        '<tr><td colspan="2" class="text-muted p-3">No alerts detected.</td></tr>'
    )
    return hw.table(rows_html), total


def render_schema_tab(split: Any, feature_names: list[str]) -> str:
    """Render the Schema tab: one row per feature with its pandas dtype.

    Args:
        split (DataSplit): Provides ``X_train`` for dtype lookup.
        feature_names (list[str]): Columns to list, in display order.

    Returns:
        str: HTML table, columns Column / Dtype.
    """

    def _dtype_str(col: str) -> str:
        dtype = str(split.X_train[col].dtype) if col in split.X_train else "unknown"
        return _html_lib.escape(dtype)

    rows = "".join(
        f"<tr><td>{_html_lib.escape(col)}</td>" f"<td><code>{_dtype_str(col)}</code></td>" f"</tr>"
        for col in feature_names
    )
    return f"""
    <div class="table-responsive">
      <table class="table table-striped">
        <thead><tr><th>Column</th><th>Dtype</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""


_TYPE_DESC = {
    "numeric": "Real number (ℝ)",
    "categorical": "Categorical",
    "boolean": "Boolean",
    "datetime": "Date",
}


def _numeric_card(
    ci: int,
    row: pd.Series,
    alerts: list[dict[str, str]],
    extreme: dict[str, list],
    clip_lower_pct: float,
    clip_upper_pct: float,
    bins: int,
    series: pd.Series,
) -> str:
    col_id = f"var-{ci}"
    btm_id = f"bottom-{ci}"
    has_null = row["missing_pct"] > 0
    left = (
        hw.stat_row("Missing", f'{row["missing_pct"]*100:.2f}%', alert=has_null)
        + hw.stat_row("Mean", f'{row["mean"]:.4g}')
        + hw.stat_row("Std", f'{row["std"]:.4g}')
    )
    right = (
        hw.stat_row("Minimum", f'{row["min"]:.4g}')
        + hw.stat_row("Maximum", f'{row["max"]:.4g}')
        + hw.stat_row("Zero %", f'{row["zero_pct"]*100:.2f}%')
    )
    hist_df, _ = hw.compute_histogram(series, bins, clip_lower_pct, clip_upper_pct)
    mini = hw.mini_hist_svg(hist_df)
    summary_cols = (
        f'<div class="col-sm-4">{hw.table(left)}</div>'
        f'<div class="col-sm-4">{hw.table(right)}</div>'
        f'<div class="col-sm-4">{mini}</div>'
    )

    q_rows = (
        hw.stat_row("Minimum", f'{row["min"]:.4g}')
        + hw.stat_row("P10", f'{row["p10"]:.4g}')
        + hw.stat_row("P25", f'{row["p25"]:.4g}')
        + hw.stat_row("Median", f'{row["p50"]:.4g}')
        + hw.stat_row("P75", f'{row["p75"]:.4g}')
        + hw.stat_row("P90", f'{row["p90"]:.4g}')
        + hw.stat_row("Maximum", f'{row["max"]:.4g}')
    )
    d_rows = (
        hw.stat_row("Std deviation", f'{row["std"]:.4g}')
        + hw.stat_row("CV", f'{row["cv"]:.3f}' if row["cv"] is not None else "—")
        + hw.stat_row("Skewness", f'{row["skewness"]:.3f}')
        + hw.stat_row("Kurtosis", f'{row["kurtosis"]:.3f}')
        + hw.stat_row("Outliers % (IQR)", f'{row["outlier_pct_iqr"]*100:.2f}%')
        + hw.stat_row("Outliers % (Z-score)", f'{row["outlier_pct_zscore"]*100:.2f}%')
    )
    stats_detail = (
        '<div class="row sub-item">'
        f'<div class="col-sm-6"><p class="h5 item-header">'
        f"Quantile statistics</p>{hw.table(q_rows)}</div>"
        f'<div class="col-sm-6"><p class="h5 item-header">'
        f"Descriptive statistics</p>{hw.table(d_rows)}</div>"
        "</div>"
    )
    dist_svg = hw.dist_hist_svg(hist_df, caption=f"Histogram — {bins} bins")
    dist_detail = (
        f'<div class="row sub-item"><div class="col-sm-12 pt-2">{dist_svg}</div></div>'
        if dist_svg
        else ""
    )

    ev = extreme.get(row["feature"], {})
    bot_rows = "".join(
        f"<tr><td>{_html_lib.escape(str(v))}</td></tr>" for v in ev.get("min_extreme", [])
    )
    top_rows = "".join(
        f"<tr><td>{_html_lib.escape(str(v))}</td></tr>" for v in ev.get("max_extreme", [])
    )
    vals_detail = (
        '<div class="row sub-item">'
        f'<div class="col-sm-6"><p class="h5 item-header">Minimum values</p>'
        f'<table class="table table-striped table-sm"><tbody>{bot_rows}</tbody></table></div>'
        f'<div class="col-sm-6"><p class="h5 item-header">Maximum values</p>'
        f'<table class="table table-striped table-sm"><tbody>{top_rows}</tbody></table></div>'
        "</div>"
    )

    return _assemble_card(
        ci,
        col_id,
        btm_id,
        row["feature"],
        _TYPE_DESC["numeric"],
        alerts,
        summary_cols,
        [
            ("Statistics", stats_detail),
            ("Distribution", dist_detail),
            ("Extreme values", vals_detail),
        ],
    )


def _categorical_card(
    ci: int, row: pd.Series, alerts: list[dict[str, str]], value_counts_df: pd.DataFrame
) -> str:
    col_id = f"var-{ci}"
    btm_id = f"bottom-{ci}"
    has_null = row["missing_pct"] > 0
    left = (
        hw.stat_row("Distinct", f'{row["n_unique"]:,}')
        + hw.stat_row("Missing", f'{row["missing_pct"]*100:.2f}%', alert=has_null)
        + hw.stat_row("Top value", str(row["top_value"]))
    )
    summary_cols = (
        f'<div class="col-sm-6">{hw.table(left)}</div>'
        f'<div class="col-sm-6">{hw.freq_bars_svg(value_counts_df, max_rows=5)}</div>'
    )
    s_rows = (
        hw.stat_row("Count", f'{row["count"]:,}')
        + hw.stat_row("Distinct", f'{row["n_unique"]:,}')
        + hw.stat_row("Top freq.", f'{row["top_freq"]*100:.2f}%')
        + hw.stat_row("High cardinality", "Yes" if row["cardinality_flag"] else "No")
    )
    stats_detail = f'<div class="row sub-item"><div class="col-sm-6">{hw.table(s_rows)}</div></div>'
    rows_html = "".join(
        f"<tr><td>{_html_lib.escape(str(r.value))}</td><td>{r.count:,}</td>"
        f"<td>{hw.bar(r.pct * 100)} {r.pct * 100:.1f}%</td></tr>"
        for r in value_counts_df.itertuples(index=False)
    )
    common_values_html = (
        '<div class="row sub-item"><div class="col-sm-12">' f"{hw.table(rows_html)}" "</div></div>"
    )
    return _assemble_card(
        ci,
        col_id,
        btm_id,
        row["feature"],
        _TYPE_DESC["categorical"],
        alerts,
        summary_cols,
        [("Statistics", stats_detail), ("Common values", common_values_html)],
    )


def _datetime_card(ci: int, row: pd.Series) -> str:
    col_id = f"var-{ci}"
    btm_id = f"bottom-{ci}"
    left = (
        hw.stat_row("Distinct", row["n_unique"])
        + hw.stat_row("Missing", f'{row["missing_pct"]*100:.2f}%', alert=row["missing_pct"] > 0)
        + hw.stat_row("Minimum", str(row["min"]))
        + hw.stat_row("Maximum", str(row["max"]))
    )
    summary_cols = f'<div class="col-sm-12">{hw.table(left)}</div>'
    return _assemble_card(
        ci, col_id, btm_id, row["feature"], _TYPE_DESC["datetime"], [], summary_cols, []
    )


def _boolean_card(ci: int, row: pd.Series) -> str:
    col_id = f"var-{ci}"
    btm_id = f"bottom-{ci}"
    left = (
        hw.stat_row("Missing", f'{row["missing_pct"]*100:.2f}%', alert=row["missing_pct"] > 0)
        + hw.stat_row("True", f'{row["n_true"]:,}')
        + hw.stat_row("False", f'{row["n_false"]:,}')
        + hw.stat_row(
            "% True", f'{row["pct_true"]*100:.2f}%' if row["pct_true"] is not None else "—"
        )
    )
    summary_cols = f'<div class="col-sm-12">{hw.table(left)}</div>'
    return _assemble_card(
        ci, col_id, btm_id, row["feature"], _TYPE_DESC["boolean"], [], summary_cols, []
    )


def _assemble_card(
    ci: int,
    col_id: str,
    btm_id: str,
    name: str,
    desc: str,
    alerts: list[dict[str, str]],
    summary_cols: str,
    detail_tabs: list[tuple[str, str]],
) -> str:
    """Wrap a summary + optional collapsible detail tabs into one variable-card row."""
    badges = hw.alert_badges(alerts)
    detail_tabs = [(label, html) for label, html in detail_tabs if html]
    more_details = ""
    if detail_tabs:
        tab_btns = "".join(
            f'<li class="nav-item" role="presentation">'
            f'<button class="nav-link{" active" if i == 0 else ""}" '
            f'data-bs-toggle="tab" data-bs-target="#pane-{ci}-{i}" role="tab" '
            f'type="button">{_html_lib.escape(label)}</button></li>'
            for i, (label, _) in enumerate(detail_tabs)
        )
        tab_panes = "".join(
            f'<div class="tab-pane fade{" show active" if i == 0 else ""}" '
            f'id="pane-{ci}-{i}" role="tabpanel">{html}</div>'
            for i, (label, html) in enumerate(detail_tabs)
        )
        more_details = f"""
            <div class="col-sm-12 text-end">
              <button class="btn btn-light btn-sm" data-bs-toggle="collapse"
                      data-bs-target="#{btm_id}" aria-expanded="false"
                      aria-controls="{btm_id}">More details</button>
            </div>
            <div class="collapse" id="{btm_id}">
              <div class="row item mt-2">
                <ul class="nav nav-tabs" role="tablist">{tab_btns}</ul>
                <div class="tab-content">{tab_panes}</div>
              </div>
            </div>"""

    return f"""
        <div class="row item">
          <div class="variable" id="{col_id}">
            <div class="row sub-item">
              <div class="col-sm-12">
                <p class="item-header h4" title="{_html_lib.escape(name)}">
                  <a href="#{col_id}">{_html_lib.escape(name)}</a><br>
                  <span class="fs-6 text-body-secondary">{desc}</span>
                </p>
                <p>{badges}</p>
              </div>
              {summary_cols}
            </div>
            {more_details}
          </div>
        </div>"""


def render_variable_cards(
    eda_report: Any,
    excel_chart_clip_lower_pct: float,
    excel_chart_clip_upper_pct: float,
    histogram_bins: int,
) -> tuple[str, dict[str, str]]:
    """Render one summary card per feature, dispatched by dtype.

    Args:
        eda_report (EDAReport): Fitted ``EDAReport`` instance.
        excel_chart_clip_lower_pct (float): Lower-tail clip fraction for
            numeric histograms — same value as ``ModelCard.excel_chart_clip_lower_pct``,
            kept identical across Excel/HTML for consistency.
        excel_chart_clip_upper_pct (float): Upper-tail clip fraction.
        histogram_bins (int): Number of histogram bins — ``settings.eda_histogram_bins``.

    Returns:
        tuple[str, dict[str, str]]: Concatenated card HTML (in numeric,
        categorical, datetime, boolean order) and a ``{feature_name:
        anchor_id}`` map for cross-linking from Alerts/Correlations/Missing
        sections.
    """
    numeric_df = eda_report.numeric_summary()
    categorical_df = eda_report.categorical_summary()
    datetime_df = eda_report.datetime_summary()
    boolean_df = eda_report.boolean_summary()
    alerts = eda_report.alerts()
    extreme = eda_report.extreme_values()

    cards: list[str] = []
    anchor_map: dict[str, str] = {}
    ci = 0

    for row in numeric_df.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        feature = row_s["feature"]
        anchor_map[feature] = f"var-{ci}"
        series = eda_report.numeric_clean_series(feature)
        cards.append(
            _numeric_card(
                ci,
                row_s,
                alerts.get(feature, []),
                extreme,
                excel_chart_clip_lower_pct,
                excel_chart_clip_upper_pct,
                histogram_bins,
                series,
            )
        )
        ci += 1

    for row in categorical_df.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        feature = row_s["feature"]
        anchor_map[feature] = f"var-{ci}"
        vc = eda_report.value_counts(feature, top_n=10)
        cards.append(_categorical_card(ci, row_s, alerts.get(feature, []), vc))
        ci += 1

    for row in datetime_df.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        anchor_map[row_s["feature"]] = f"var-{ci}"
        cards.append(_datetime_card(ci, row_s))
        ci += 1

    for row in boolean_df.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        anchor_map[row_s["feature"]] = f"var-{ci}"
        cards.append(_boolean_card(ci, row_s))
        ci += 1

    return "".join(cards), anchor_map


def render_correlations_section(eda_report: Any, feature_anchor: dict[str, str]) -> str:
    """Render the Pearson correlation matrix as a sticky-header heatmap table.

    Args:
        eda_report (EDAReport): Fitted ``EDAReport`` instance.
        feature_anchor (dict[str, str]): Mapping of feature name to variable
            card anchor id, from ``render_variable_cards()``.

    Returns:
        str: HTML for the correlations table, or ``""`` when fewer than 2
        numeric columns were present at fit time.
    """
    corr_df = eda_report.correlation_table()
    if corr_df.empty:
        return ""
    cols = list(corr_df.columns)
    th_corner = (
        'style="position:sticky;top:0;left:0;z-index:3;background:#fff;'
        'border-right:1px solid #dee2e6;border-bottom:1px solid #dee2e6"'
    )
    th_top = (
        'style="position:sticky;top:0;z-index:2;background:#fff;font-size:10px;'
        "max-width:70px;overflow:hidden;writing-mode:vertical-lr;transform:rotate(180deg);"
        'padding:4px 2px;border-bottom:1px solid #dee2e6"'
    )
    td_left = (
        'style="position:sticky;left:0;z-index:1;background:#fff;font-size:11px;'
        'padding:4px 8px;white-space:nowrap;border-right:1px solid #dee2e6"'
    )
    hdr_ths = "".join(f"<th {th_top}>{_html_lib.escape(c[:14])}</th>" for c in cols)
    body_rows = []
    for c1 in cols:
        cells = []
        for c2 in cols:
            val = corr_df.loc[c1, c2]
            val = None if pd.isna(val) else round(float(val), 3)
            title_text = f'title="{_html_lib.escape(c1)} × {_html_lib.escape(c2)}"'
            val_str = val if val is not None else "&mdash;"
            cells.append(
                f'<td style="background:{hw.corr_color(val)};color:white;'
                f'text-align:center;font-size:10px;padding:4px 6px" {title_text}>'
                f"{val_str}</td>"
            )
        anchor = feature_anchor.get(c1)
        label = (
            f'<a href="#{_html_lib.escape(anchor)}"><b>{_html_lib.escape(c1[:16])}</b></a>'
            if anchor
            else f"<b>{_html_lib.escape(c1[:16])}</b>"
        )
        body_rows.append(f"<tr><td {td_left}>{label}</td>" + "".join(cells) + "</tr>")
    table_style = "border-collapse:separate;border-spacing:0"
    return f"""
    <div style="overflow:auto;max-height:560px;position:relative">
      <table class="table table-bordered table-sm" style="{table_style}">
        <thead><tr><th {th_corner}></th>{hdr_ths}</tr></thead>
        <tbody>{"".join(body_rows)}</tbody>
      </table>
    </div>"""


def render_missing_section(eda_report: Any, feature_anchor: dict[str, str]) -> str:
    """Render the Missing values tab-pair: a count table and a bar chart.

    Args:
        eda_report (EDAReport): Fitted ``EDAReport`` instance.
        feature_anchor (dict[str, str]): Mapping of feature name to variable
            card anchor id.

    Returns:
        str: HTML with two Bootstrap tabs (Count / Bar chart).
    """
    total_rows = eda_report.overview_summary()["total_rows"]
    numeric_df = eda_report.numeric_summary()[["feature", "missing_pct", "count"]]
    categorical_df = eda_report.categorical_summary()[["feature", "missing_pct", "count"]]
    combined = pd.concat([numeric_df, categorical_df], ignore_index=True)
    combined = combined.sort_values("missing_pct", ascending=False)

    rows = []
    bar_rows: list[tuple[str, float, int]] = []
    for r in combined.itertuples(index=False):
        pct = r.missing_pct * 100
        if pct <= 0:
            continue
        null_count = total_rows - r.count
        anchor = feature_anchor.get(r.feature)
        label = (
            f'<a href="#{_html_lib.escape(anchor)}"><code>{_html_lib.escape(r.feature)}</code></a>'
            if anchor
            else f"<code>{_html_lib.escape(r.feature)}</code>"
        )
        rows.append(
            f"<tr><td>{label}</td><td>{pct:.2f}%</td>"
            f'<td>{hw.bar(pct, color="#dc3545")}</td></tr>'
        )
        bar_rows.append((r.feature, pct, null_count))

    rows_html = (
        "".join(rows)
        or '<tr><td colspan="3" class="text-muted p-3">No missing values &#x2713;</td></tr>'
    )
    count_tab = f"""
    <table class="table table-striped">
      <thead><tr><th>Column</th><th>Missing %</th><th></th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>"""
    bar_tab = f'<div class="mt-3 pb-4">{hw.missing_bar_svg(bar_rows)}</div>'

    return f"""
    <ul class="nav nav-tabs" role="tablist">
      <li class="nav-item" role="presentation">
        <button class="nav-link active" data-bs-toggle="tab" data-bs-target="#pane-miss-count"
                role="tab" type="button">Count</button></li>
      <li class="nav-item" role="presentation">
        <button class="nav-link" data-bs-toggle="tab" data-bs-target="#pane-miss-bar"
                role="tab" type="button">Bar chart</button></li>
    </ul>
    <div class="tab-content">
      <div class="tab-pane fade show active" id="pane-miss-count" role="tabpanel">{count_tab}</div>
      <div class="tab-pane fade" id="pane-miss-bar" role="tabpanel">{bar_tab}</div>
    </div>"""


def render_sample_section(eda_report: Any) -> str:
    """Render sample head/tail rows and duplicate-row content, gated by their compliance opt-ins.

    Args:
        eda_report (EDAReport): Fitted ``EDAReport`` instance.

    Returns:
        str: HTML table(s), or ``""`` when neither ``include_sample_rows``
        nor ``include_duplicate_row_content`` was enabled at construction.
    """
    sample = eda_report.sample_rows()
    duplicate_df = eda_report.duplicate_row_content()
    if sample["head"].empty and duplicate_df.empty:
        return ""

    parts = []
    if not sample["head"].empty:
        cols = list(sample["head"].columns)
        hdr = "".join(f"<th>{_html_lib.escape(c)}</th>" for c in cols)
        combined = pd.concat([sample["head"], sample["tail"]], ignore_index=True)
        body = "".join(
            "<tr>" + "".join(f"<td>{_html_lib.escape(str(v))}</td>" for v in row) + "</tr>"
            for row in combined.itertuples(index=False)
        )
        parts.append(
            f"""
        <p class="h5 item-header">Sample rows (head + tail)</p>
        <div class="table-responsive">
          <table class="table table-striped table-sm"><thead><tr>{hdr}</tr></thead>
          <tbody>{body}</tbody></table>
        </div>"""
        )
    if not duplicate_df.empty:
        cols = list(duplicate_df.columns)
        hdr = "".join(f"<th>{_html_lib.escape(c)}</th>" for c in cols)
        body = "".join(
            "<tr>" + "".join(f"<td>{_html_lib.escape(str(v))}</td>" for v in row) + "</tr>"
            for row in duplicate_df.itertuples(index=False)
        )
        parts.append(
            f"""
        <p class="h5 item-header mt-3">Duplicate rows</p>
        <div class="table-responsive">
          <table class="table table-striped table-sm"><thead><tr>{hdr}</tr></thead>
          <tbody>{body}</tbody></table>
        </div>"""
        )
    return "".join(parts)
