"""Tests for dscompanion.docs.html_widgets — HTML/SVG rendering primitives."""

from __future__ import annotations

import pandas as pd

from dscompanion.docs import html_widgets as hw


def test_stat_row_escapes_and_formats_value():
    row = hw.stat_row("Missing", "12,345")
    assert "<th>Missing</th>" in row
    assert "12,345" in row


def test_stat_row_none_value_renders_em_dash():
    assert ">—<" in hw.stat_row("Mean", None)


def test_stat_row_escapes_html_in_value():
    row = hw.stat_row("Top value", "<script>")
    assert "<script>" not in row
    assert "&lt;script&gt;" in row


def test_table_wraps_rows_in_table_striped():
    html = hw.table("<tr><td>x</td></tr>")
    assert 'class="table table-striped"' in html
    assert "<tr><td>x</td></tr>" in html


def test_bar_clamps_above_100_percent():
    html = hw.bar(150.0)
    assert "width:100.0%" in html


def test_corr_color_none_is_neutral():
    assert hw.corr_color(None) == "#ddd"


def test_corr_color_positive_and_negative_differ():
    assert hw.corr_color(0.9) != hw.corr_color(-0.9)


def test_alert_badges_empty_list_returns_empty_string():
    assert hw.alert_badges([]) == ""


def test_alert_badges_renders_one_badge_per_alert():
    alerts = [{"type": "HIGH_MISSING", "detail": "40% missing"}]
    html = hw.alert_badges(alerts)
    assert html.count("badge") == 1
    assert "HIGH_MISSING" in html
    assert "40% missing" in html


def test_compute_histogram_returns_empty_for_short_series():
    hist_df, has_kde = hw.compute_histogram(
        pd.Series([1.0]), bins=10, clip_lower_pct=0.0, clip_upper_pct=0.0
    )
    assert hist_df.empty
    assert has_kde is False


def test_compute_histogram_returns_bin_center_count_kde_columns():
    series = pd.Series(range(100), dtype=float)
    hist_df, has_kde = hw.compute_histogram(series, bins=10, clip_lower_pct=0.0, clip_upper_pct=0.0)
    assert list(hist_df.columns) == ["bin_center", "count", "kde"]
    assert has_kde is True
    assert hist_df["count"].sum() == 100


def test_mini_hist_svg_empty_for_all_zero_counts():
    hist_df = pd.DataFrame({"bin_center": [1, 2], "count": [0, 0], "kde": [0.0, 0.0]})
    assert hw.mini_hist_svg(hist_df) == ""


def test_mini_hist_svg_renders_svg_for_nonzero_counts():
    hist_df = pd.DataFrame({"bin_center": [1, 2, 3], "count": [5, 10, 2], "kde": [0.0, 0.0, 0.0]})
    svg = hw.mini_hist_svg(hist_df)
    assert svg.startswith("<svg")
    assert svg.count("<rect") == 3


def test_dist_hist_svg_includes_caption():
    hist_df = pd.DataFrame({"bin_center": [1, 2, 3], "count": [5, 10, 2], "kde": [0.0, 0.0, 0.0]})
    svg = hw.dist_hist_svg(hist_df, caption="Histogram — 3 bins")
    assert "Histogram — 3 bins" in svg


def test_freq_bars_svg_empty_dataframe_returns_empty_string():
    assert hw.freq_bars_svg(pd.DataFrame(columns=["value", "count", "pct"])) == ""


def test_freq_bars_svg_renders_one_row_per_value():
    vc_df = pd.DataFrame({"value": ["a", "b"], "count": [10, 5], "pct": [0.66, 0.33]})
    svg = hw.freq_bars_svg(vc_df)
    assert svg.count("<rect") == 2


def test_missing_bar_svg_no_missing_renders_checkmark_message():
    assert "No missing values" in hw.missing_bar_svg([])


def test_missing_bar_svg_renders_one_bar_per_column():
    rows = [("col_a", 40.0, 400), ("col_b", 10.0, 100)]
    svg = hw.missing_bar_svg(rows)
    assert svg.count("<rect") == 2


def test_combo_bar_line_svg_renders_bars_and_polyline():
    svg = hw.combo_bar_line_svg(
        categories=["1", "2", "3"],
        bar_values=[0.1, 0.2, 0.15],
        bar_label="event_rate",
        line_values=[1.0, 1.8, 1.5],
        line_label="cumulative_lift",
        title="Decile",
    )
    assert svg.count("<rect") == 3
    assert "<polyline" in svg
    assert "Decile" in svg


def test_horizontal_bar_svg_renders_one_bar_per_label():
    svg = hw.horizontal_bar_svg(labels=["f1", "f2"], values=[0.5, 0.2], title="SHAP")
    assert svg.count("<rect") == 2
    assert "SHAP" in svg


def test_horizontal_bar_svg_empty_values_returns_empty_string():
    assert hw.horizontal_bar_svg(labels=[], values=[], title="x") == ""


def test_ensure_bootstrap_populates_cache_or_falls_back():
    hw.ensure_bootstrap()
    # Either the fetch succeeded (non-empty CSS) or failed gracefully (empty
    # string, caller falls back to CDN <link> tags) — both are valid outcomes
    # in a sandboxed/offline test environment, so only assert no exception
    # and that the cache key exists.
    assert "css" in hw._bootstrap_cache
