"""Interactive mode — Step 4: Explore the Data (EDA).

Read-only — no confirm-before-apply gate is needed since nothing here
mutates the data (design rule #1 in streamlit.md gates mutations; there are
none in this step). Runs ``dscompanion.eda.EDAReport`` directly against the
Step-3-confirmed split and walks through four sequential, gated sub-checks
(design rule: show one category of finding at a time, not everything at
once): Missing Values, Univariate EDA (stats tables + a per-column
histogram/KDE chart), Bivariate EDA — Feature vs Target (IV-ranked table
+ a per-feature target-rate-by-bin chart), and Multivariate EDA —
Feature vs Feature (Pearson clustered heatmap, VIF table, Cramér's V for
categorical pairs). Outliers/Zeros/Skewness/Other Findings screens are
still deferred.

Local-dev Streamlit UI only — never imported by dscompanion's runtime code.
"""

from __future__ import annotations

import dataclasses
import logging

import pandas as pd
import streamlit as st
from interactive_state import add_audit_entry, dark_fig, set_step_confirmed, set_step_status

from dscompanion.config import settings
from dscompanion.eda import EDAReport
from dscompanion.eda.report import EDAAlert
from dscompanion.split import DataSplit

logger = logging.getLogger(__name__)

__all__ = ["render_step_eda"]

# Mirrors dscompanion/docs/model_card.py's _ALERT_BADGE_COLOR semantics, translated to
# st.badge's fixed named palette (red/orange/yellow/blue/green/violet/gray/primary —
# no arbitrary hex support, unlike model_card.py's raw HTML badges).
_EDA_ALERT_BADGE_COLOR: dict[str, str] = {
    EDAAlert.HIGH_MISSING: "orange",
    EDAAlert.CONSTANT: "gray",
    EDAAlert.NEAR_ZERO_VARIANCE: "gray",
    EDAAlert.HIGH_CARDINALITY: "orange",
    EDAAlert.SKEWED: "yellow",
    EDAAlert.ZEROS: "blue",
    EDAAlert.LOW_IV: "red",
    EDAAlert.MULTICOLLINEAR: "violet",
    EDAAlert.HIGH_CORRELATION: "violet",
    EDAAlert.IMBALANCED: "green",
}
_DEFAULT_EDA_ALERT_COLOR = "gray"


def _render_overview(report: EDAReport) -> None:
    """Show row/column/missing/duplicate counts as quick-glance metrics.

    Args:
        report (EDAReport): A fitted (``run_all()``-called) report.

    Returns:
        None
    """
    overview = report.overview_summary()
    cols = st.columns(4)
    cols[0].metric("Rows", f"{overview['total_rows']:,}")
    cols[1].metric("Columns", f"{overview['total_columns']:,}")
    # overview_summary()'s missing_pct/duplicate_pct are already 0-100 scale
    # (computed as round(100 * x / y, 2)) — unlike numeric_summary()'s
    # missing_pct, which is a 0-1 fraction. Format as a plain number, not
    # Python's ":.1%" spec (that would multiply by 100 a second time).
    cols[2].metric("Missing", f"{overview['missing_pct']:.1f}%")
    cols[3].metric("Duplicate rows", f"{overview['duplicate_pct']:.1f}%")


def _render_missing_values(report: EDAReport) -> None:
    """Render the Missing Values check: bar chart, nullity correlation, and HIGH_MISSING alerts.

    Args:
        report (EDAReport): A fitted (``run_all()``-called) report.

    Returns:
        None
    """
    # theme=None: render dscompanion's own apply_dscompanion_theme() styling as
    # authored — Streamlit's default theme="streamlit" re-skins
    # fonts/colors to match the page theme (dark mode washes out dscompanion's
    # dark-on-white text to near-invisible light grey).
    st.plotly_chart(dark_fig(report.missing_values_chart()), width="stretch", theme=None)

    corr_fig = report.missing_correlation_heatmap()
    if corr_fig.data:
        st.plotly_chart(dark_fig(corr_fig), width="stretch", theme=None)
        st.caption(
            "Columns that tend to go missing together score close to 1, often the same "
            "underlying cause (for example, a field only collected for a subset of customers)."
        )

    alerts = report.alerts()
    high_missing_alerts = [
        (feature, alert)
        for feature, feature_alerts in alerts.items()
        for alert in feature_alerts
        if alert["type"] == "HIGH_MISSING"
    ]
    if not high_missing_alerts:
        return

    st.warning(f"Found {len(high_missing_alerts)} column(s) with too much missing data.")
    for feature, alert in high_missing_alerts:
        st.info(
            f"Column '{feature}' is missing in {alert['detail']} of rows, too many gaps to "
            "be reliable. May be removed automatically in Step 6 (Feature Selection)."
        )


def _render_numeric_column_detail(row: pd.Series, alerts: list[dict[str, str]]) -> None:
    """Render alert badges and a Quantile/Descriptive statistics panel for one numeric column.

    Mirrors a ydata-profiling-style per-column detail panel — pulls entirely from
    already-computed data (``row`` from ``numeric_summary()``, ``alerts`` from
    ``EDAReport.alerts()``), no new statistics are computed here.

    Args:
        row (pd.Series): The selected column's row from ``numeric_summary()``.
        alerts (list[dict[str, str]]): This column's alerts from
            ``EDAReport.alerts()`` — each a ``{"type": str, "detail": str}``
            dict. Empty list when the column has none.

    Returns:
        None
    """
    for alert in alerts:
        st.badge(
            alert["type"],
            color=_EDA_ALERT_BADGE_COLOR.get(alert["type"], _DEFAULT_EDA_ALERT_COLOR),
            help=alert["detail"],
        )

    iqr = row["p75"] - row["p25"]
    quantile_rows = [
        {"Statistic": "Minimum", "Value": row["min"]},
        {"Statistic": "P1", "Value": row["p1"]},
        {"Statistic": "P2", "Value": row["p2"]},
        {"Statistic": "P3", "Value": row["p3"]},
        {"Statistic": "P4", "Value": row["p4"]},
        {"Statistic": "P5", "Value": row["p5"]},
        {"Statistic": "P10", "Value": row["p10"]},
        {"Statistic": "Q1 (P25)", "Value": row["p25"]},
        {"Statistic": "Median (P50)", "Value": row["p50"]},
        {"Statistic": "Q3 (P75)", "Value": row["p75"]},
        {"Statistic": "P90", "Value": row["p90"]},
        {"Statistic": "P95", "Value": row["p95"]},
        {"Statistic": "P96", "Value": row["p96"]},
        {"Statistic": "P97", "Value": row["p97"]},
        {"Statistic": "P98", "Value": row["p98"]},
        {"Statistic": "P99", "Value": row["p99"]},
        {"Statistic": "Maximum", "Value": row["max"]},
        {"Statistic": "IQR", "Value": iqr},
    ]
    descriptive_rows = [
        {"Statistic": "Mean", "Value": row["mean"]},
        {"Statistic": "Std deviation", "Value": row["std"]},
        {"Statistic": "CV", "Value": row["cv"]},
        {"Statistic": "Skewness", "Value": row["skewness"]},
        {"Statistic": "Kurtosis", "Value": row["kurtosis"]},
        {"Statistic": "Missing %", "Value": row["missing_pct"] * 100},
        {"Statistic": "Zero %", "Value": row["zero_pct"] * 100},
        {"Statistic": "Outliers (IQR) %", "Value": row["outlier_pct_iqr"] * 100},
        {"Statistic": "Outliers (Z-score) %", "Value": row["outlier_pct_zscore"] * 100},
    ]

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Quantile statistics**")
        st.dataframe(
            pd.DataFrame(quantile_rows).round({"Value": 4}), width="stretch", hide_index=True
        )
    with col_right:
        st.markdown("**Descriptive statistics**")
        st.dataframe(
            pd.DataFrame(descriptive_rows).round({"Value": 4}), width="stretch", hide_index=True
        )


def _render_numeric_univariate(
    report: EDAReport, numeric_summary: pd.DataFrame, near_zero_variance_threshold: float
) -> None:
    """Render the numeric stats table, its flag-threshold legend, and the per-column KDE chart.

    Args:
        report (EDAReport): A fitted (``run_all()``-called) report.
        numeric_summary (pd.DataFrame): ``report.numeric_summary()``'s
            output — passed in rather than recomputed, since the caller
            already needs it to decide whether this section renders at all.
        near_zero_variance_threshold (float): The value actually used to
            compute this report's ``near_zero_variance_flag`` column (the
            user's slider choice, not necessarily dscompanion's default) — shown
            here so the legend never drifts from what the table reflects.

    Returns:
        None
    """
    st.markdown("**Numeric Columns**")
    st.dataframe(numeric_summary, width="stretch")
    with st.expander("What do outlier_pct_*, constant_flag, and near_zero_variance_flag mean?"):
        st.markdown(
            f"- **outlier_pct_iqr**: share of values far outside the typical range, below "
            f"Q1 or above Q3 by more than {settings.iqr_multiplier}× the interquartile range "
            "(a standard statistical rule of thumb). Informational only; doesn't currently "
            "drive any automatic feature removal.\n"
            f"- **outlier_pct_zscore**: share of values more than "
            f"{settings.zscore_outlier_threshold} standard deviations from the mean. Also "
            "informational only.\n"
            "- **constant_flag**: every row has the same value (or the column is entirely "
            "missing), carries no information for modelling. **Suggested for removal** in "
            "Step 5 below.\n"
            f"- **near_zero_variance_flag**: values barely vary (variance below "
            f"{near_zero_variance_threshold}, set via the slider above), almost constant, "
            "unlikely to help the model. **Suggested for removal** in Step 5 below; raise "
            "or lower the slider to change which columns get flagged."
        )

    numeric_cols = numeric_summary["feature"].tolist()
    col_select, col_toggle = st.columns([2, 1])
    with col_select:
        selected_col = st.selectbox(
            "Pick a numeric column to inspect", numeric_cols, key="int.eda.kde_column"
        )
    lower_pctile = int(settings.eda_chart_clip_lower_pct * 100)
    upper_pctile = int(100 - settings.eda_chart_clip_upper_pct * 100)
    view_original = "Original"
    view_trimmed = f"Outlier-Treated (P{lower_pctile}–P{upper_pctile})"
    with col_toggle:
        chart_view = st.segmented_control(
            "Chart view",
            options=[view_original, view_trimmed],
            default=view_original,
            key="int.eda.kde_chart_view",
            help="Switch between the raw distribution and a P%d–P%d trimmed "
            "view, useful for highly skewed columns where a few extreme values "
            "otherwise hide the bulk of the distribution. Trims charting only, "
            "never the underlying data." % (lower_pctile, upper_pctile),
        )
    exclude_outliers = chart_view == view_trimmed
    st.plotly_chart(
        dark_fig(
            report.numeric_distribution_with_kde(selected_col, exclude_outliers=exclude_outliers)
        ),
        width="stretch",
        theme=None,
    )
    st.caption(
        "Blue bars: how often values in this range occur (density scale). Red line: a "
        "smoothed estimate of the underlying distribution shape (KDE)."
    )

    selected_row = numeric_summary.set_index("feature").loc[selected_col]
    selected_alerts = report.alerts().get(selected_col, [])
    _render_numeric_column_detail(selected_row, selected_alerts)


def _render_categorical_univariate(report: EDAReport, categorical_summary: pd.DataFrame) -> None:
    """Render the categorical stats table, cardinality legend, and a per-column bar chart.

    Args:
        report (EDAReport): A fitted (``run_all()``-called) report — used to
            fetch value counts for the selected column.
        categorical_summary (pd.DataFrame): ``report.categorical_summary()``'s output.

    Returns:
        None
    """
    import plotly.graph_objects as go

    st.markdown("**Categorical Columns**")
    st.dataframe(categorical_summary, width="stretch")
    st.caption(
        f"`cardinality_flag` is True when a column has more than "
        f"{settings.high_cardinality_threshold} unique values. Too many distinct categories "
        "can cause problems for some encoders and may need special handling."
    )

    cat_cols = categorical_summary["feature"].tolist()
    selected_cat = st.selectbox(
        "Pick a categorical column to inspect", cat_cols, key="int.eda.cat_column"
    )
    counts = report.categorical_value_counts(selected_cat)
    fig = go.Figure(
        go.Bar(
            x=counts.index.astype(str).tolist(),
            y=counts.values.tolist(),
            # Single-series bar chart — one consistent hue, no legend needed
            # (dataviz skill's nominal-categorical rule). Categorical
            # palette slot 1, dark step (matches interactive_state.py's
            # _PALETTE_DARK[0], since this chart goes through dark_fig()
            # below and marker_color bypasses colorway).
            marker_color="#3987e5",
        )
    )
    fig.update_layout(
        xaxis_title=selected_cat,
        yaxis_title="Count",
        height=420,
        margin={"t": 30, "b": 160, "l": 100, "r": 20},
    )
    # update_xaxes / update_yaxes always merge — called after dark_fig so
    # gridcolor/linecolor survive and tick settings are never overwritten.
    dark_fig(fig)
    fig.update_xaxes(tickangle=-45, automargin=True)
    fig.update_yaxes(tickformat=",d", automargin=True)
    st.plotly_chart(fig, width="stretch", theme=None)
    st.caption(
        "Bar height = number of rows with that value. "
        "Only the top categories are shown for high-cardinality columns."
    )


def _render_univariate_eda(report: EDAReport, near_zero_variance_threshold: float) -> bool:
    """Render the Univariate EDA check: numeric stats/chart first, categorical stats after.

    Gated like Step 4's own sub-checks — numeric columns (table, threshold
    legend, per-column KDE chart) are reviewed and confirmed before
    categorical columns appear, per the same "one thing at a time"
    principle already applied to Missing Values vs. Univariate EDA.

    Args:
        report (EDAReport): A fitted (``run_all()``-called) report.
        near_zero_variance_threshold (float): The threshold actually used
            to build ``report`` — passed through to the numeric section's
            legend so it never drifts from what the table reflects.

    Returns:
        bool: ``True`` once both the numeric and categorical sections (or
        whichever are non-empty) have been shown — i.e. this check is
        complete and the caller may render its own "Continue" button.
        ``False`` while still waiting on the numeric-to-categorical gate.
    """
    numeric_summary = report.numeric_summary()
    categorical_summary = report.categorical_summary()

    if numeric_summary.empty:
        numeric_reviewed = True
    else:
        _render_numeric_univariate(report, numeric_summary, near_zero_variance_threshold)
        numeric_reviewed = st.session_state.get("int.eda.univariate_numeric_done", False)
        if not numeric_reviewed:
            if st.button("Continue to categorical columns", key="int.eda.continue_to_categorical"):
                st.session_state["int.eda.univariate_numeric_done"] = True
                st.rerun()
            return False

    if categorical_summary.empty:
        if numeric_summary.empty:
            st.caption("No numeric or categorical columns to review.")
        else:
            st.caption("No categorical columns in this dataset.")
        return True

    if not numeric_summary.empty:
        st.divider()
    _render_categorical_univariate(report, categorical_summary)
    return True


_IV_LEAKAGE_POWER = "Very Strong"
_IV_LEAKAGE_MESSAGE = "Suspicious, Check for Data Leakage"
_IV_GLOSS = {
    "Useless": "tells the model almost nothing about the target on its own.",
    "Weak": "has a weak relationship with the target.",
    "Medium": "has a moderate, useful relationship with the target.",
    "Strong": "has a strong relationship with the target.",
    _IV_LEAKAGE_POWER: f"{_IV_LEAKAGE_MESSAGE}. IV above 0.5 usually means a feature is "
    "unusually strongly related to the target, which can happen if it accidentally contains "
    "the answer.",
}


def _render_bivariate_eda(report: EDAReport) -> None:
    """Render the Bivariate EDA check: an IV-ranked table plus a per-feature target-rate chart.

    Args:
        report (EDAReport): A fitted (``run_all()``-called) report.

    Returns:
        None
    """
    st.caption(
        "**Information Value (IV)** measures how much a feature, on its own, helps tell the "
        "two target classes apart. Below 0.02 is considered not useful; above 0.5 is unusually "
        "strong (worth checking the feature isn't accidentally leaking the answer)."
    )
    iv_table = report.iv_table()
    display_iv = iv_table.copy()
    display_iv["predictive_power"] = display_iv["predictive_power"].replace(
        _IV_LEAKAGE_POWER, _IV_LEAKAGE_MESSAGE
    )
    st.dataframe(display_iv, width="stretch")

    leaking = iv_table.loc[iv_table["predictive_power"] == _IV_LEAKAGE_POWER, "feature"].tolist()
    if leaking:
        st.warning(
            f"{_IV_LEAKAGE_MESSAGE}: {', '.join(leaking)}. IV above 0.5 with the target. "
            "This can happen if a feature accidentally contains the answer (for example, it "
            "was derived from the target, or only exists after the outcome is known)."
        )

    features = iv_table["feature"].tolist()
    selected_feature = st.selectbox(
        "Pick a feature to inspect", features, key="int.eda.bivariate_feature"
    )
    power = iv_table.set_index("feature").loc[selected_feature, "predictive_power"]
    gloss = _IV_GLOSS.get(power, "")
    st.caption(f"**{selected_feature}**, predictive power: **{power}**. {gloss}")
    st.plotly_chart(
        dark_fig(report.target_rate_by_bin_chart(selected_feature)), width="stretch", theme=None
    )
    st.caption(
        "Each bar is a group of similar values for this feature. A clear upward or downward "
        "trend across bars means the feature relates to the target; a flat line means it "
        "doesn't, regardless of how it looks in isolation."
    )
    if power in ("Useless", "Weak"):
        st.caption(
            f"'{selected_feature}' has {power.lower()} predictive power and may be suggested "
            "for removal during Feature Selection (Step 6)."
        )


def _render_multivariate_eda(report: EDAReport) -> None:
    """Render the Multivariate EDA check: Pearson heatmap, VIF table, and Cramér's V table.

    Numeric section (heatmap + VIF) renders when at least two numeric columns
    were present at fit time.  Categorical section (Cramér's V) renders when
    at least two categorical columns were present.  When neither condition
    holds, a short explanatory caption is shown instead.

    Args:
        report (EDAReport): A fitted (``run_all()``-called) report.

    Returns:
        None
    """
    numeric_done = False
    try:
        heatmap = report.pearson_heatmap()
        st.markdown("**Pearson correlation (clustered)**")
        st.plotly_chart(dark_fig(heatmap), width="stretch", theme=None)
        st.caption(
            "Columns are reordered so that highly correlated groups cluster together. "
            "Deep red means strong positive correlation; deep blue means strong negative "
            "correlation. Features that are very similar to each other (r close to ±1) "
            "may carry redundant information."
        )

        high_corr = report.flag_high_correlation()
        if not high_corr.empty:
            st.warning(
                f"{len(high_corr)} pair(s) exceed the correlation threshold "
                f"({settings.correlation_threshold}). Highly correlated pairs carry "
                "overlapping information; keeping both columns adds noise without "
                "adding predictive power."
            )
            st.dataframe(high_corr, width="stretch")
            st.caption(
                "`pearson_r` close to 1.0 means both columns move together; "
                "close to -1.0 means they move in opposite directions. "
                "Either way, one column can likely substitute for both."
            )
        else:
            st.success(
                f"No pairs exceed the correlation threshold "
                f"({settings.correlation_threshold}). No obviously redundant numeric "
                "columns found."
            )

        vif = report.vif_table()
        flagged = vif.loc[vif["flag"], "feature"].tolist()
        st.divider()
        st.markdown("**Variance Inflation Factor (VIF)**")
        with st.expander("What is VIF?"):
            st.markdown(
                f"VIF measures how much a feature's variance is inflated by its correlation "
                f"with other features. Features with VIF above "
                f"**{settings.vif_threshold}** are flagged.\n\n"
                "**Rule of thumb:** VIF > 5, monitor. VIF > 10, consider removing one "
                "of the correlated features or using regularisation."
            )
        if flagged:
            st.warning(
                f"High VIF (possible multicollinearity): **{', '.join(flagged)}**. "
                "These features may be too similar to each other. Regularised models "
                "(Ridge, SAGA logistic) handle this gracefully; tree models are unaffected."
            )
        else:
            st.success("No features flagged for multicollinearity (all VIF within threshold).")
        with st.expander("Show full VIF table"):
            st.dataframe(vif, width="stretch")
        numeric_done = True
    except ValueError:
        pass  # fewer than 2 numeric columns — skip silently

    cramers = report.cramers_v_table()
    if not cramers.empty:
        if numeric_done:
            st.divider()
        st.markdown("**Cramér's V: Categorical Associations**")
        st.dataframe(cramers, width="stretch")
        st.caption(
            "Cramér's V measures how strongly two categorical columns are associated "
            "(0 = independent, 1 = perfectly associated). Pairs above ~0.6 may be "
            "near-redundant. This is the categorical equivalent of the Pearson heatmap above."
        )
    elif not numeric_done:
        st.caption(
            "No numeric or categorical column pairs to analyse for feature-feature relationships."
        )


def _exclude_identifier_columns(split: DataSplit, identifier_columns: list[str]) -> DataSplit:
    """Returns a copy of ``split`` with identifier columns dropped from every X partition.

    Identifier columns (e.g. UCIC ID, account number) chosen at Step 2 are
    never useful for EDA — analyzing them wastes screen space at best and
    triggers spurious HIGH_CARDINALITY/leakage-style alerts at worst. App-layer
    transformation only; ``dscompanion.split.DataSplit`` itself is untouched.

    Args:
        split (DataSplit): Split confirmed at Step 3.
        identifier_columns (list[str]): Columns to drop from
            ``train_X``/``val_X``/``test_X``/``oot_X``. No-op when empty.

    Returns:
        DataSplit: A new ``DataSplit`` (``train_y``/``val_y``/``test_y``/
        ``oot_y``/``metadata`` unchanged) with identifier columns absent from
        every feature partition. Returns ``split`` itself, unmodified, when
        ``identifier_columns`` is empty.
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


def render_step_eda(split: DataSplit, target: str) -> bool:
    """Render Step 4 (EDA) and report whether it's confirmed.

    Args:
        split (DataSplit): Split confirmed at Step 3 — EDA runs on
            ``split.train_X``/``split.train_y`` only, mirroring
            ``PipelineRunner``'s own ``_run_eda()`` calling convention.
            Identifier columns chosen at Step 2 are dropped before EDA runs
            (see ``_exclude_identifier_columns``).
        target (str): Confirmed target column name.

    Returns:
        bool: ``True`` once the user clicks "Continue" (with or without
        running the check). ``False`` while still reviewing.
    """
    st.subheader("Step 4: Explore the data (EDA)")

    identifier_columns = st.session_state.get("interactive.identifier_columns") or []
    split = _exclude_identifier_columns(split, identifier_columns)
    if identifier_columns:
        st.caption(
            f"Excluded {len(identifier_columns)} identifier column(s) from this check: "
            f"{', '.join(identifier_columns)}."
        )

    run_eda = st.toggle(
        "Run a data quality check before training? (recommended)",
        value=True,
        key="int.eda.run_eda",
    )
    st.caption(
        "This looks for issues like missing values, constant columns, or features that look "
        "suspiciously identical to the target, before you spend time training."
    )

    if not run_eda:
        if st.button("Continue", key="int.eda.continue_skip"):
            st.session_state["interactive.eda_report"] = None
            set_step_confirmed("eda", True)
            set_step_status("eda", "flagged")
            add_audit_entry("eda", "Skipped the data quality check.")
        return st.session_state["interactive.step_confirmed"]["eda"]

    with st.expander(":material/tune: Adjust data-quality thresholds", expanded=False):
        high_missing_threshold = st.slider(
            "High-missing threshold",
            0.05,
            0.95,
            settings.high_missing_threshold,
            0.05,
            key="int.eda.high_missing_threshold",
            help="Missing-value rate above which a column is flagged as high-missing.",
        )
        near_zero_variance_threshold = st.slider(
            "Near-zero-variance threshold",
            0.0,
            0.5,
            settings.near_zero_variance_threshold,
            0.01,
            key="int.eda.near_zero_variance_threshold",
            help="Variance below which a numeric column is flagged as near-constant.",
        )

    thresholds = (high_missing_threshold, near_zero_variance_threshold)
    if st.session_state.get("int.eda._cached_thresholds") != thresholds:
        st.session_state.pop("int.eda._cached_report", None)
        st.session_state.pop("int.eda.missing_values_done", None)
        st.session_state.pop("int.eda.univariate_done", None)
        st.session_state.pop("int.eda.bivariate_done", None)
        st.session_state["int.eda._cached_thresholds"] = thresholds

    st.markdown("**Check 1 of 4 This Round: Missing Values**")

    if "int.eda._cached_report" not in st.session_state:
        try:
            report = EDAReport(
                split=split,
                target=target,
                high_missing_threshold=high_missing_threshold,
                near_zero_variance_threshold=near_zero_variance_threshold,
            ).run_all()
            st.session_state["int.eda._cached_report"] = report
        except Exception as exc:
            logger.exception("Interactive mode — EDA failed")
            st.error(f"Couldn't run the data quality check: {exc}")
            if st.button(
                "Continue without the data quality check", key="int.eda.continue_after_error"
            ):
                st.session_state["interactive.eda_report"] = None
                set_step_confirmed("eda", True)
                set_step_status("eda", "flagged")
                add_audit_entry("eda", "Data quality check failed. Skipped.")
            return st.session_state["interactive.step_confirmed"]["eda"]

    report = st.session_state["int.eda._cached_report"]
    _render_overview(report)
    _render_missing_values(report)

    if not st.session_state.get("int.eda.missing_values_done"):
        if st.button("Continue to univariate EDA", key="int.eda.continue_to_univariate"):
            st.session_state["int.eda.missing_values_done"] = True
            st.rerun()
        return st.session_state["interactive.step_confirmed"]["eda"]

    st.divider()
    st.markdown("**Check 2 of 4 This Round: Univariate EDA**")
    if not _render_univariate_eda(report, near_zero_variance_threshold):
        return st.session_state["interactive.step_confirmed"]["eda"]

    if not st.session_state.get("int.eda.univariate_done"):
        if st.button("Continue to bivariate EDA", key="int.eda.continue_to_bivariate"):
            st.session_state["int.eda.univariate_done"] = True
            st.rerun()
        return st.session_state["interactive.step_confirmed"]["eda"]

    st.divider()
    st.markdown("**Check 3 of 4 This Round: Bivariate EDA, Feature vs Target**")
    _render_bivariate_eda(report)

    if not st.session_state.get("int.eda.bivariate_done"):
        if st.button("Continue to multivariate EDA", key="int.eda.continue_to_multivariate"):
            st.session_state["int.eda.bivariate_done"] = True
            st.rerun()
        return st.session_state["interactive.step_confirmed"]["eda"]

    st.divider()
    st.markdown("**Check 4 of 4 This Round: Multivariate EDA, Feature vs Feature**")
    _render_multivariate_eda(report)

    if st.button("Continue", key="int.eda.continue"):
        n_alerts = sum(
            1
            for feature_alerts in report.alerts().values()
            for alert in feature_alerts
            if alert["type"] == "HIGH_MISSING"
        )
        st.session_state["interactive.eda_report"] = report
        st.session_state["interactive.eda_thresholds"] = {
            "high_missing": high_missing_threshold,
            "near_zero_variance": near_zero_variance_threshold,
        }
        set_step_confirmed("eda", True)
        set_step_status("eda", "done")
        if n_alerts:
            add_audit_entry("eda", f"Missing Values check: {n_alerts} column(s) flagged.")
        else:
            add_audit_entry("eda", "Missing Values check: no issues found.")
        add_audit_entry("eda", "Univariate EDA check: reviewed stats and distribution charts.")
        add_audit_entry(
            "eda", "Bivariate EDA check: reviewed feature-vs-target IV and rate charts."
        )
        add_audit_entry(
            "eda",
            "Multivariate EDA check: reviewed Pearson heatmap, flagged pairs, VIF, and Cramér's V.",
        )

    return st.session_state["interactive.step_confirmed"]["eda"]
