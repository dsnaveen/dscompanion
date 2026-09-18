"""HTML/SVG rendering primitives shared by every ModelCard HTML section renderer.

Ported from spark_data_profiler.py's rendering helpers
(https://github.com/dsnaveen/spark-data-profiling) — same visual output, adapted to
take plain pandas/dict inputs instead of a Spark-derived report dict. No
Jinja2, no Plotly, no JS chart library: every chart is a hand-built inline
SVG string, and the only external dependency is a one-time Bootstrap 5
CDN fetch, embedded inline for a self-contained report.
"""

from __future__ import annotations

import html as _html_lib
import logging
from typing import Any

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

__all__ = [
    "ALERT_BADGE_COLOR",
    "DEFAULT_ALERT_COLOR",
    "ensure_bootstrap",
    "bootstrap_css",
    "bootstrap_js",
    "stat_row",
    "table",
    "bar",
    "corr_color",
    "alert_badges",
    "compute_histogram",
    "mini_hist_svg",
    "dist_hist_svg",
    "freq_bars_svg",
    "missing_bar_svg",
    "combo_bar_line_svg",
    "horizontal_bar_svg",
]

# Alert-type -> Bootstrap badge colour. Generic (not EDA-specific): any
# section producing {"type": str, "detail": str} alert dicts can reuse
# alert_badges(). Unknown types fall back to DEFAULT_ALERT_COLOR so new
# alert types introduced elsewhere never raise here.
ALERT_BADGE_COLOR: dict[str, str] = {
    "HIGH_MISSING": "#e67e22",
    "CONSTANT": "#7f8c8d",
    "NEAR_ZERO_VARIANCE": "#95a5a6",
    "HIGH_CARDINALITY": "#e67e22",
    "SKEWED": "#f39c12",
    "ZEROS": "#3498db",
    "LOW_IV": "#c0392b",
    "MULTICOLLINEAR": "#8e44ad",
    "HIGH_CORRELATION": "#8e44ad",
    "IMBALANCED": "#16a085",
}
DEFAULT_ALERT_COLOR = "#888888"

# Bootstrap 5 self-contained assets — fetched once per process and embedded
# inline so the rendered HTML stays a single offline-viewable file, even
# though generating it needs outbound internet access. Falls back to CDN
# <link>/<script> tags (degraded: needs internet at *view* time instead) if
# the fetch fails.
_BOOTSTRAP_CSS_URL = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
_BOOTSTRAP_JS_URL = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"
_bootstrap_cache: dict[str, str] = {}


def _fetch_url(url: str, timeout: int = 15) -> str:
    """Fetch ``url`` and return its decoded text, or an empty string on any error.

    Args:
        url (str): URL to fetch.
        timeout (int): Socket timeout in seconds. Defaults to 15.

    Returns:
        str: Decoded response body, or ``""`` if the request fails for any
        reason (no network, timeout, non-2xx status).
    """
    import urllib.request

    try:
        # nosec B310 -- url is never caller/user-controlled: the only two call sites
        # below pass hardcoded HTTPS CDN constants, never external input, so this
        # isn't the SSRF/arbitrary-scheme risk this rule generally guards against.
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # nosec B310
            return resp.read().decode("utf-8")
    except Exception as exc:
        logger.warning("Bootstrap asset fetch failed for %s: %s", url, type(exc).__name__)
        return ""


def ensure_bootstrap() -> None:
    """Populate the module-level Bootstrap CSS/JS cache on first call.

    Fetches Bootstrap 5 CSS and JS from a CDN once per process. If the
    fetch fails the cache values stay empty and callers fall back to CDN
    ``<link>``/``<script>`` tags instead of inline-embedded assets.

    Args:
        None

    Returns:
        None
    """
    if "css" in _bootstrap_cache:
        return
    logger.info("Fetching Bootstrap 5 for self-contained model card report")
    _bootstrap_cache["css"] = _fetch_url(_BOOTSTRAP_CSS_URL)
    _bootstrap_cache["js"] = _fetch_url(_BOOTSTRAP_JS_URL)
    status = "embedded" if _bootstrap_cache["css"] else "CDN fallback (offline?)"
    logger.info("Bootstrap assets: %s", status)


def bootstrap_css() -> str:
    """Return a ``<style>``/``<link>`` tag for Bootstrap 5 CSS.

    Args:
        None

    Returns:
        str: Inline ``<style>{css}</style>`` when the fetch in
        ``ensure_bootstrap()`` succeeded, else a CDN ``<link>`` tag.
    """
    css = _bootstrap_cache.get("css", "")
    if css:
        return f"<style>{css}</style>"
    return (
        '<link rel="stylesheet" '
        'href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">'
    )


def bootstrap_js() -> str:
    """Return a ``<script>`` tag for Bootstrap 5 JS (bundle, includes Popper).

    Args:
        None

    Returns:
        str: Inline ``<script>{js}</script>`` when the fetch succeeded, else
        a CDN ``<script src=...>`` tag.
    """
    js = _bootstrap_cache.get("js", "")
    if js:
        return f"<script>{js}</script>"
    return (
        '<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/'
        'dist/js/bootstrap.bundle.min.js"></script>'
    )


def stat_row(label: str, value: Any, alert: bool = False) -> str:
    """Render one ``<tr><th>label</th><td>value</td></tr>`` stat-table row.

    Args:
        label (str): Row label, rendered unescaped (always a hardcoded
            string from the caller, never user/data-derived).
        value (Any): Value to display. ``None`` renders as an em-dash.
            Converted to ``str`` and HTML-escaped.
        alert (bool): When ``True``, applies the ``alert-info`` CSS class
            to flag this row (e.g. a nonzero missing count). Defaults to
            ``False``.

    Returns:
        str: One ``<tr>`` HTML row.
    """
    td_class = ' class="alert-info"' if alert else ""
    v = "—" if value is None else _html_lib.escape(str(value))
    return f'<tr{td_class}><th>{label}</th><td style="white-space:nowrap">{v}</td></tr>'


def table(rows_html: str) -> str:
    """Wrap pre-rendered ``<tr>`` rows in a responsive Bootstrap striped table.

    Args:
        rows_html (str): Concatenated ``<tr>...</tr>`` row markup.

    Returns:
        str: Full ``<div class="table-responsive"><table>...</table></div>``.
    """
    return (
        f'<div class="table-responsive"><table class="table table-striped">'
        f"<tbody>{rows_html}</tbody></table></div>"
    )


def bar(pct: float, color: str = "#0d6efd") -> str:
    """Render an inline horizontal progress bar.

    Args:
        pct (float): Fill percentage in ``[0, 100]``. Values above 100 are
            clamped.
        color (str): CSS colour for the filled portion. Defaults to
            ``"#0d6efd"`` (Bootstrap primary blue).

    Returns:
        str: HTML string containing the bar markup.
    """
    clamped = min(pct, 100.0)
    return (
        f'<div style="background:#eee;border-radius:3px;height:7px;width:100%;margin-top:4px">'
        f'<div style="background:{color};width:{clamped:.1f}%;height:7px;'
        f'border-radius:3px"></div></div>'
    )


def corr_color(value: float | None) -> str:
    """Map a correlation coefficient to an RGB background colour for the heatmap.

    Args:
        value (float | None): Correlation coefficient in ``[-1.0, 1.0]``.

    Returns:
        str: CSS ``rgb(...)`` colour string, or ``"#ddd"`` for ``None``.
    """
    if value is None:
        return "#ddd"
    v = float(value)
    if v >= 0:
        r = int(255 * (1 - v))
        return f"rgb({r},100,220)"
    r = int(255 * (1 + v))
    return f"rgb(220,80,{r})"


def alert_badges(alerts: list[dict[str, str]]) -> str:
    """Render a list of alert dicts as inline Bootstrap badge spans.

    Args:
        alerts (list[dict[str, str]]): List of ``{"type": str, "detail":
            str}`` dicts, typically from ``EDAReport.alerts()``.

    Returns:
        str: Concatenated badge spans, or ``""`` if ``alerts`` is empty.
    """

    def _badge_html(a: dict[str, str]) -> str:
        bg_color = ALERT_BADGE_COLOR.get(a["type"], DEFAULT_ALERT_COLOR)
        title_text = _html_lib.escape(a["detail"])
        type_text = _html_lib.escape(a["type"])
        return (
            f'<span class="badge" style="background:{bg_color}" '
            f'title="{title_text}">{type_text}</span>'
        )

    return "".join(_badge_html(a) for a in alerts)


def compute_histogram(
    series: pd.Series, bins: int, clip_lower_pct: float, clip_upper_pct: float
) -> tuple[pd.DataFrame, bool]:
    """Compute histogram bins plus a count-scaled KDE curve, where computable.

    Shared by ``ModelCard.to_excel()`` (native xlsxwriter chart) and
    ``ModelCard.to_html()`` (inline SVG) — one histogram implementation for
    both output formats. Clips extreme values first, purely for chart
    readability — the clipped values are never written anywhere or used
    outside the chart.

    Args:
        series (pd.Series): Clean (already dropna'd) numeric values.
        bins (int): Number of histogram bins.
        clip_lower_pct (float): Lower-tail fraction clipped before binning.
        clip_upper_pct (float): Upper-tail fraction clipped before binning.

    Returns:
        tuple[pd.DataFrame, bool]: A ``bin_center, count, kde`` table
        (``kde`` is all-zero when not computable) and a flag for whether
        the KDE curve is meaningful. Returns an empty DataFrame and
        ``False`` when ``series`` has fewer than 2 values.
    """
    if len(series) < 2:
        return pd.DataFrame(columns=["bin_center", "count", "kde"]), False

    clean = series
    if clip_lower_pct > 0 or clip_upper_pct > 0:
        lower = clean.quantile(clip_lower_pct)
        upper = clean.quantile(1 - clip_upper_pct)
        clean = clean.clip(lower, upper)

    counts, edges = np.histogram(clean, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    has_kde = clean.nunique() >= 2
    kde_values = np.zeros(len(centers))
    if has_kde:
        bin_width = edges[1] - edges[0]
        kde_values = gaussian_kde(clean)(centers) * len(clean) * bin_width

    hist_df = pd.DataFrame({"bin_center": centers, "count": counts, "kde": kde_values})
    return hist_df, has_kde


def mini_hist_svg(hist_df: pd.DataFrame) -> str:
    """Render a small inline histogram SVG for a per-feature summary card.

    Args:
        hist_df (pd.DataFrame): Output of ``compute_histogram`` — columns
            ``bin_center``, ``count``.

    Returns:
        str: Self-contained ``<svg>`` markup, or ``""`` when ``hist_df`` is
        empty or every bin count is zero.
    """
    if hist_df.empty or hist_df["count"].max() == 0:
        return ""
    counts = hist_df["count"].tolist()
    max_c = max(counts)
    n = len(counts)
    vw, vh, pad = 216, 90, 8
    bar_w = (vw - 2 * pad) / n
    parts = [
        f'<line x1="{pad}" y1="{vh-12}" x2="{vw-pad}" y2="{vh-12}" '
        f'stroke="#dee2e6" stroke-width="1"/>'
    ]
    for i, cnt in enumerate(counts):
        bh = max(2, int(60 * cnt / max_c))
        x = pad + i * bar_w
        y = (vh - 12) - bh
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(1, bar_w-1):.1f}" '
            f'height="{bh}" fill="#0d6efd" rx="1"/>'
        )
    return (
        f'<svg class="img-responsive" viewBox="0 0 {vw} {vh}" '
        f'style="width:100%;max-width:{vw}px;display:block;margin:auto" '
        f'xmlns="http://www.w3.org/2000/svg">' + "".join(parts) + "</svg>"
    )


def dist_hist_svg(hist_df: pd.DataFrame, caption: str = "") -> str:
    """Render a full-size histogram SVG, with an optional KDE line overlay.

    Args:
        hist_df (pd.DataFrame): Output of ``compute_histogram`` — columns
            ``bin_center``, ``count``, ``kde``.
        caption (str): Optional caption shown under the chart. Defaults to
            ``""`` (no caption).

    Returns:
        str: A ``<figure>`` containing the SVG (and caption, if given), or
        ``""`` when ``hist_df`` is empty or every bin count is zero.
    """
    if hist_df.empty or hist_df["count"].max() == 0:
        return ""
    counts = hist_df["count"].tolist()
    centers = hist_df["bin_center"].tolist()
    kde = hist_df["kde"].tolist() if "kde" in hist_df.columns else []
    has_kde = bool(kde) and max(kde) > 0

    n = len(counts)
    max_c = max(counts)
    max_y = max(max_c, max(kde) if has_kde else 0)
    pad_l, pad_r, pad_t, pad_b = 58, 16, 16, 46
    svg_w = 600
    chart_h = 180
    svg_h = chart_h + pad_t + pad_b
    usable = svg_w - pad_l - pad_r
    bar_w = usable / n

    parts = []
    for lvl in range(6):
        frac = lvl / 5
        y = pad_t + chart_h - int(chart_h * frac)
        dash = 'stroke-dasharray="4,4"' if lvl > 0 else ""
        parts.append(
            f'<line x1="{pad_l}" y1="{y}" x2="{svg_w-pad_r}" y2="{y}" '
            f'stroke="#dee2e6" stroke-width="1" {dash}/>'
            f'<text x="{pad_l-5}" y="{y+4}" font-size="9" fill="#6c757d" '
            f'text-anchor="end">{int(max_y*frac):,}</text>'
        )
    for i, cnt in enumerate(counts):
        bh = max(1, int(chart_h * cnt / max_y)) if max_y else 0
        x = pad_l + i * bar_w
        y = pad_t + chart_h - bh
        tip = f"{centers[i]:.4g}: {cnt:,}"
        parts.append(
            f'<rect x="{x+1:.1f}" y="{y:.1f}" width="{max(1, bar_w-2):.1f}" '
            f'height="{bh}" fill="#0d6efd" rx="1"><title>{_html_lib.escape(tip)}</title></rect>'
        )
    if has_kde:
        points = []
        for i, k in enumerate(kde):
            x = pad_l + (i + 0.5) * bar_w
            y = pad_t + chart_h - (chart_h * k / max_y if max_y else 0)
            points.append(f"{x:.1f},{y:.1f}")
        parts.append(
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="#dc3545" stroke-width="2"/>'
        )
    shown, idxs = set(), [0, n // 4, n // 2, 3 * n // 4, n - 1]
    for idx in idxs:
        if idx in shown or idx >= n:
            continue
        shown.add(idx)
        x = pad_l + (idx + 0.5) * bar_w
        parts.append(
            f'<text x="{x:.1f}" y="{pad_t+chart_h+16}" font-size="9" '
            f'fill="#495057" text-anchor="middle">'
            f'{_html_lib.escape(f"{centers[idx]:.4g}")}</text>'
        )

    _cap = _html_lib.escape(caption)
    return (
        f'<figure style="margin:0">'
        f'<svg viewBox="0 0 {svg_w} {svg_h}" '
        f'style="width:100%;max-width:{svg_w}px;display:block;margin:auto" '
        f'xmlns="http://www.w3.org/2000/svg">'
        + "".join(parts)
        + "</svg>"
        + (
            f'<figcaption class="text-center text-body-secondary small mt-1">{_cap}</figcaption>'
            if _cap
            else ""
        )
        + "</figure>"
    )


def freq_bars_svg(vc_df: pd.DataFrame, max_rows: int = 10) -> str:
    """Render a horizontal value-frequency bar chart for a categorical/boolean feature.

    Args:
        vc_df (pd.DataFrame): Columns ``value``, ``count``, ``pct`` (``pct``
            as a fraction 0-1), typically from ``EDAReport.value_counts()``.
        max_rows (int): Maximum number of values to draw. Defaults to 10.

    Returns:
        str: Self-contained ``<svg>`` markup, or ``""`` when ``vc_df`` is
        empty.
    """
    if vc_df.empty:
        return ""
    rows = vc_df.head(max_rows)
    max_c = rows["count"].max()
    bar_h = 22
    pad_l, pad_r, pad_t = 140, 60, 6
    svg_w = 600
    usable = svg_w - pad_l - pad_r
    svg_h = len(rows) * (bar_h + 4) + pad_t + 10
    parts = []
    for i, row in enumerate(rows.itertuples(index=False)):
        bw = max(2, int(usable * row.count / max_c)) if max_c else 0
        y = pad_t + i * (bar_h + 4)
        raw_value = str(row.value)[:22] if row.value is not None else "(null)"
        label = _html_lib.escape(raw_value)
        tip = f"{raw_value}: {row.count:,} ({row.pct * 100:.1f}%)"
        parts.append(
            f'<text x="{pad_l-6}" y="{y+bar_h//2+4}" font-size="11" '
            f'fill="#495057" text-anchor="end">{label}</text>'
            f'<rect x="{pad_l}" y="{y}" width="{bw}" height="{bar_h}" '
            f'fill="#0d6efd" rx="2"><title>{_html_lib.escape(tip)}</title></rect>'
            f'<text x="{pad_l+bw+4}" y="{y+bar_h//2+4}" font-size="10" '
            f'fill="#6c757d">{row.count:,} ({row.pct*100:.1f}%)</text>'
        )
    return (
        f'<svg viewBox="0 0 {svg_w} {svg_h}" style="width:100%;max-width:{svg_w}px" '
        f'xmlns="http://www.w3.org/2000/svg">' + "".join(parts) + "</svg>"
    )


def missing_bar_svg(missing_rows: list[tuple[str, float, int]]) -> str:
    """Render a bar chart of missing % per column, for columns with any nulls.

    Args:
        missing_rows (list[tuple[str, float, int]]): ``(column, missing_pct,
            missing_count)`` tuples, ``missing_pct`` in ``[0, 100]``. Only
            rows with ``missing_count > 0`` should be passed in.

    Returns:
        str: Self-contained ``<svg>`` markup, or a "no missing values"
        message when ``missing_rows`` is empty.
    """
    if not missing_rows:
        return '<p class="text-muted p-3">No missing values &#x2713;</p>'
    n = len(missing_rows)
    bar_w = max(24, min(60, 600 // n))
    gap = 4
    left, top, chart_h = 42, 10, 200
    axis_h = chart_h - 40
    total_w = left + n * (bar_w + gap) + 10

    parts = []
    for pct in (0, 25, 50, 75, 100):
        y = top + axis_h - int(axis_h * pct / 100)
        parts.append(
            f'<line x1="{left}" y1="{y}" x2="{total_w}" y2="{y}" '
            f'stroke="#dee2e6" stroke-width="1"/>'
            f'<text x="{left-4}" y="{y+3}" font-size="9" fill="#6c757d" '
            f'text-anchor="end">{pct}%</text>'
        )
    for i, (col, pct, cnt) in enumerate(missing_rows):
        bh = max(2, int(axis_h * pct / 100))
        x = left + i * (bar_w + gap)
        y = top + axis_h - bh
        lbl = _html_lib.escape(col[:12] + ("…" if len(col) > 12 else ""))
        cx = x + bar_w // 2
        parts.append(
            f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bh}" fill="#dc3545" rx="2">'
            f"<title>{_html_lib.escape(col)}: {pct:.1f}% missing ({cnt:,} rows)</title></rect>"
            f'<text transform="rotate(-45,{cx},{top+axis_h+6})" x="{cx}" y="{top+axis_h+6}" '
            f'font-size="9" fill="#495057" text-anchor="end">{lbl}</text>'
        )
    return (
        f'<svg viewBox="0 0 {total_w} {chart_h}" '
        f'style="width:100%;max-width:{total_w}px;overflow:visible" '
        f'xmlns="http://www.w3.org/2000/svg">' + "".join(parts) + "</svg>"
    )


def combo_bar_line_svg(
    categories: list[str],
    bar_values: list[float],
    bar_label: str,
    line_values: list[float],
    line_label: str,
    title: str = "",
) -> str:
    """Render a combo bar+line chart (e.g. decile event rate + cumulative lift).

    Bars use the left axis (scaled to ``max(bar_values)``); the line uses
    an independent right axis (scaled to ``max(line_values)``) — mirrors
    ``ModelCard._add_decile_chart()``'s Excel combo chart semantics.

    Args:
        categories (list[str]): X-axis category labels (e.g. decile numbers).
        bar_values (list[float]): One bar height per category.
        bar_label (str): Legend label for the bar series.
        line_values (list[float]): One line-point height per category, same
            length as ``categories``.
        line_label (str): Legend label for the line series.
        title (str): Chart title, shown above the plot. Defaults to ``""``.

    Returns:
        str: Self-contained ``<svg>`` markup, or ``""`` when ``categories``
        is empty.
    """
    if not categories:
        return ""
    n = len(categories)
    pad_l, pad_r, pad_t, pad_b = 50, 50, 44, 40
    svg_w, chart_h = 600, 200
    svg_h = chart_h + pad_t + pad_b
    usable = svg_w - pad_l - pad_r
    bar_w = usable / n
    max_bar = max(bar_values) or 1
    max_line = max(line_values) or 1

    parts = []
    if title:
        parts.append(
            f'<text x="{svg_w/2}" y="14" font-size="12" fill="#212529" '
            f'text-anchor="middle" font-weight="600">{_html_lib.escape(title)}</text>'
        )
    for i, (cat, bv) in enumerate(zip(categories, bar_values)):
        bh = max(1, int(chart_h * bv / max_bar))
        x = pad_l + i * bar_w
        y = pad_t + chart_h - bh
        parts.append(
            f'<rect x="{x+2:.1f}" y="{y:.1f}" width="{max(1, bar_w-4):.1f}" '
            f'height="{bh}" fill="#0d6efd" rx="1">'
            f"<title>{_html_lib.escape(bar_label)}: {bv:.4g}</title></rect>"
        )
        cx = x + bar_w / 2
        parts.append(
            f'<text x="{cx:.1f}" y="{pad_t+chart_h+14}" font-size="9" '
            f'fill="#495057" text-anchor="middle">{_html_lib.escape(str(cat))}</text>'
        )
    points = []
    for i, lv in enumerate(line_values):
        x = pad_l + (i + 0.5) * bar_w
        y = pad_t + chart_h - (chart_h * lv / max_line)
        points.append(f"{x:.1f},{y:.1f}")
    parts.append(
        f'<polyline points="{" ".join(points)}" fill="none" stroke="#dc3545" stroke-width="2"/>'
    )
    for i, lv in enumerate(line_values):
        x = pad_l + (i + 0.5) * bar_w
        y = pad_t + chart_h - (chart_h * lv / max_line)
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#dc3545">'
            f"<title>{_html_lib.escape(line_label)}: {lv:.4g}</title></circle>"
        )
    bar_label_text = _html_lib.escape(bar_label)
    line_label_text = _html_lib.escape(line_label)
    parts.append(
        f'<text x="{pad_l}" y="30" font-size="10" fill="#0d6efd">'
        f"■ {bar_label_text}</text>"
        f'<text x="{pad_l+120}" y="30" font-size="10" fill="#dc3545">'
        f"● {line_label_text}</text>"
    )
    return (
        f'<svg viewBox="0 0 {svg_w} {svg_h}" style="width:100%;max-width:{svg_w}px" '
        f'xmlns="http://www.w3.org/2000/svg">' + "".join(parts) + "</svg>"
    )


def horizontal_bar_svg(labels: list[str], values: list[float], title: str = "") -> str:
    """Render a generic ranked horizontal bar chart (SHAP importance, PSI, leaderboard metric).

    Args:
        labels (list[str]): One label per bar, in display order (already
            sorted by the caller — this function does not sort).
        values (list[float]): One value per bar, same length as ``labels``.
        title (str): Chart title, shown above the plot. Defaults to ``""``.

    Returns:
        str: Self-contained ``<svg>`` markup, or ``""`` when ``labels`` is
        empty.
    """
    if not labels:
        return ""
    max_v = max(values) or 1
    bar_h = 22
    pad_l, pad_r, pad_t = 160, 60, 24 if title else 6
    svg_w = 600
    usable = svg_w - pad_l - pad_r
    svg_h = len(labels) * (bar_h + 4) + pad_t + 10
    parts = []
    if title:
        parts.append(
            f'<text x="{svg_w/2}" y="14" font-size="12" fill="#212529" '
            f'text-anchor="middle" font-weight="600">{_html_lib.escape(title)}</text>'
        )
    for i, (label, value) in enumerate(zip(labels, values)):
        bw = max(2, int(usable * value / max_v)) if max_v else 0
        y = pad_t + i * (bar_h + 4)
        lbl = _html_lib.escape(str(label)[:24])
        parts.append(
            f'<text x="{pad_l-6}" y="{y+bar_h//2+4}" font-size="11" '
            f'fill="#495057" text-anchor="end">{lbl}</text>'
            f'<rect x="{pad_l}" y="{y}" width="{bw}" height="{bar_h}" fill="#0d6efd" rx="2">'
            f"<title>{lbl}: {value:.4g}</title></rect>"
            f'<text x="{pad_l+bw+4}" y="{y+bar_h//2+4}" font-size="10" '
            f'fill="#6c757d">{value:.4g}</text>'
        )
    return (
        f'<svg viewBox="0 0 {svg_w} {svg_h}" style="width:100%;max-width:{svg_w}px" '
        f'xmlns="http://www.w3.org/2000/svg">' + "".join(parts) + "</svg>"
    )
