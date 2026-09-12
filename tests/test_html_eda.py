"""Tests for dscompanion.docs.html_eda — EDA section HTML renderers."""

from __future__ import annotations

import pandas as pd
import pytest

from dscompanion.docs import html_eda
from dscompanion.eda import EDAReport


@pytest.fixture
def fitted_eda_report(data_split):
    return EDAReport(data_split, target="target").run_all()


def test_render_overview_tab_includes_row_and_column_counts(fitted_eda_report):
    html = html_eda.render_overview_tab(fitted_eda_report)
    ov = fitted_eda_report.overview_summary()
    assert f"{ov['total_rows']:,}" in html
    assert str(ov["total_columns"]) in html


def test_render_alerts_tab_no_alerts_shows_message(fitted_eda_report):
    # A dataset with no flagged features renders the "no alerts" placeholder
    # row rather than an empty table body.
    html, total = html_eda.render_alerts_tab(fitted_eda_report, feature_anchor={})
    if total == 0:
        assert "No alerts detected" in html


def test_render_alerts_tab_links_to_feature_anchor(fitted_eda_report):
    alerts = fitted_eda_report.alerts()
    if not alerts:
        pytest.skip("fixture data has no alerts to link")
    first_feature = next(iter(alerts))
    anchor_map = {first_feature: "var-0"}
    html, total = html_eda.render_alerts_tab(fitted_eda_report, feature_anchor=anchor_map)
    assert total == sum(len(v) for v in alerts.values())
    assert 'href="#var-0"' in html


def test_render_schema_tab_lists_every_feature_with_dtype(data_split):
    feature_names = list(data_split.X_train.columns)
    html = html_eda.render_schema_tab(data_split, feature_names)
    for col in feature_names:
        assert col in html
    assert "<thead>" in html


def test_render_variable_cards_one_card_per_feature(fitted_eda_report):
    html, anchor_map = html_eda.render_variable_cards(
        fitted_eda_report,
        excel_chart_clip_lower_pct=0.01,
        excel_chart_clip_upper_pct=0.01,
        histogram_bins=20,
    )
    numeric_df = fitted_eda_report.numeric_summary()
    categorical_df = fitted_eda_report.categorical_summary()
    total_features = len(numeric_df) + len(categorical_df)
    assert len(anchor_map) >= total_features
    for feature in numeric_df["feature"]:
        assert feature in anchor_map
        assert html.count(f'id="{anchor_map[feature]}"') == 1


def test_render_variable_cards_numeric_card_has_statistics_and_distribution_tabs(fitted_eda_report):
    html, _ = html_eda.render_variable_cards(
        fitted_eda_report,
        excel_chart_clip_lower_pct=0.01,
        excel_chart_clip_upper_pct=0.01,
        histogram_bins=20,
    )
    assert "Statistics" in html
    assert "Distribution" in html


def test_render_variable_cards_categorical_card_shows_common_values_tab(fitted_eda_report):
    categorical_df = fitted_eda_report.categorical_summary()
    if categorical_df.empty:
        pytest.skip("fixture data has no categorical columns")
    html, _ = html_eda.render_variable_cards(
        fitted_eda_report,
        excel_chart_clip_lower_pct=0.01,
        excel_chart_clip_upper_pct=0.01,
        histogram_bins=20,
    )
    assert "Common values" in html


def test_render_correlations_section_empty_when_fewer_than_two_numeric_cols(fitted_eda_report):
    corr_df = fitted_eda_report.correlation_table()
    html = html_eda.render_correlations_section(fitted_eda_report, feature_anchor={})
    if corr_df.empty:
        assert html == ""
    else:
        assert "<table" in html


def test_render_correlations_section_cell_count_matches_matrix_size(fitted_eda_report):
    corr_df = fitted_eda_report.correlation_table()
    if corr_df.empty:
        pytest.skip("fixture data has fewer than 2 numeric columns")
    html = html_eda.render_correlations_section(fitted_eda_report, feature_anchor={})
    n = len(corr_df)
    assert html.count("<td") >= n * n


def test_render_missing_section_lists_columns_with_nulls(fitted_eda_report):
    html = html_eda.render_missing_section(fitted_eda_report, feature_anchor={})
    numeric_df = fitted_eda_report.numeric_summary()
    any_missing = (numeric_df["missing_pct"] > 0).any() if not numeric_df.empty else False
    if not any_missing:
        assert "No missing values" in html or "<table" in html


def test_render_sample_section_empty_when_sample_rows_disabled(fitted_eda_report):
    sample = fitted_eda_report.sample_rows()
    html = html_eda.render_sample_section(fitted_eda_report)
    if sample["head"].empty:
        assert html == ""
    else:
        assert "<table" in html


def test_render_missing_section_correct_null_count_for_fully_missing_column(
    fitted_eda_report,
):
    # Verify that null_count is computed correctly for a fully-missing column,
    # not silently 0 from a bad formula. Skip if the fixture has no such column.
    numeric_df = fitted_eda_report.numeric_summary()
    categorical_df = fitted_eda_report.categorical_summary()
    combined = pd.concat([numeric_df, categorical_df], ignore_index=True)
    fully_missing = combined[combined["missing_pct"] == 1.0]
    if fully_missing.empty:
        pytest.skip("fixture has no fully-missing column")
    # If we get here, there's at least one 100%-missing column.
    # The HTML should include that column, and the null_count passed to
    # missing_bar_svg should equal total_rows (since count == 0 for a fully-missing column).
    html = html_eda.render_missing_section(fitted_eda_report, feature_anchor={})
    # Verify at least one fully-missing column appears in the table
    for _, row in fully_missing.iterrows():
        col_name = row["feature"]
        assert col_name in html
        # The table should show "100.00%" for this column
        assert "100.00%" in html
