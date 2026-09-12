"""Common Plotly utilities shared across the dscompanion package."""

from __future__ import annotations

import base64

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dscompanion.config import settings

__all__ = ["apply_dscompanion_theme", "fig_to_base64", "render_corr_heatmap"]

# Categorical palette (light-mode steps) — 8 hues, fixed order, never cycled.
# Validated (CVD-safe adjacent pairs, contrast, lightness band) by the dataviz
# skill's own reference palette; see interactive_state.py's _PALETTE_DARK for
# the matching dark-mode steps applied on top by dark_fig().
_PALETTE = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]


def apply_dscompanion_theme(fig: "go.Figure") -> "go.Figure":
    """Apply the standard dscompanion colour palette, typography, and background
    styling to a Plotly figure by calling ``fig.update_layout`` in-place,
    ensuring visual consistency across all charts produced by the package.

    The theme sets a system-sans font stack, a near-white paper background,
    a light plot background, the dscompanion 8-colour categorical palette,
    standardised margins (with ``automargin`` enabled on both axes, so long
    tick labels — e.g. a horizontal bar chart's category names — expand the
    margin instead of being clipped), a semi-transparent legend box, and
    light grid lines on both axes. This is the **light-mode** theme — used
    directly by
    ``dscompanion/docs/model_card.py``'s static HTML reports and every chart in
    ``dscompanion/eda/``, ``dscompanion/explain/``, ``dscompanion/calibration/``,
    ``dscompanion/models/``, and ``dscompanion/selection/``. ``dscompanion/app/``'s
    Streamlit UI additionally layers ``interactive_state.py``'s ``dark_fig()``
    on top for its dark display — this function itself must stay
    light-mode-appropriate for the non-Streamlit consumers above.

    Args:
        fig (go.Figure): A Plotly ``Figure`` instance to style.  The figure
            is mutated directly; no copy is made.

    Returns:
        go.Figure: The same ``Figure`` object passed in, with the dscompanion
        layout applied.  The return value and the input reference point to
        the same object.
    """
    fig.update_layout(
        font={
            "family": 'system-ui, -apple-system, "Segoe UI", sans-serif',
            "size": 13,
            "color": "#0b0b0b",
        },
        paper_bgcolor="#fcfcfb",
        plot_bgcolor="#f9f9f7",
        colorway=_PALETTE,
        margin={"l": 60, "r": 30, "t": 50, "b": 60},
        legend={"bgcolor": "rgba(255,255,255,0.8)", "bordercolor": "#c3c2b7", "borderwidth": 1},
        xaxis={"gridcolor": "#e1e0d9", "linecolor": "#c3c2b7", "automargin": True},
        yaxis={"gridcolor": "#e1e0d9", "linecolor": "#c3c2b7", "automargin": True},
    )
    return fig


def fig_to_base64(fig: "go.Figure", format: str = "png", scale: float = 1.5) -> str:
    """Render a Plotly figure to an image and return it as a plain
    base64-encoded string suitable for embedding in HTML ``<img>`` tags or
    python-docx documents, without any ``data:`` URI prefix.

    For SVG output the ``scale`` parameter is ignored because SVG is a
    vector format.  Raster export (PNG) requires the ``kaleido`` package to
    be installed; an ``ValueError`` is raised by Plotly if ``kaleido`` is
    absent.

    Args:
        fig (go.Figure): A Plotly ``Figure`` instance to render.
        format (str): Output image format.  Either ``"png"`` (raster,
            default) or ``"svg"`` (vector).  Passing any other string will
            cause Plotly to raise a ``ValueError``.
        scale (float): Pixel density multiplier applied to raster output
            only.  ``1.5`` (default) produces 1.5× the logical pixel
            dimensions, yielding sharper images on high-DPI displays.
            Has no effect when ``format="svg"``.

    Returns:
        str: Base64-encoded image data as a UTF-8 string with no leading
        ``data:image/...;base64,`` prefix.  Never returns an empty string
        for a valid figure; raises on rendering failure.
    """
    if format == "svg":
        img_bytes = fig.to_image(format="svg")
    else:
        img_bytes = fig.to_image(format="png", scale=scale)
    return base64.b64encode(img_bytes).decode("utf-8")


def render_corr_heatmap(matrix: pd.DataFrame, title: str) -> "go.Figure":
    """Render a square correlation-style matrix as a themed, annotated Plotly heatmap.

    Shared by ``BivariateAnalyser.correlation_heatmap`` (Pearson correlation
    between features) and ``MissingnessAnalyser.nullity_correlation_heatmap``
    (phi-coefficient correlation between columns' missingness indicators) —
    both need identical layout/colorscale/annotation logic over a square
    ``[-1, 1]``-ranged matrix, so this is factored out rather than duplicated.

    Args:
        matrix (pd.DataFrame): Square DataFrame with matching row/column
            labels and values in ``[-1, 1]`` (e.g. a ``.corr()`` result).
        title (str): Plot title.

    Returns:
        go.Figure: Annotated heatmap with ``RdBu`` colour scale centred at
        zero, themed via ``apply_dscompanion_theme``.
    """
    fig = go.Figure(
        go.Heatmap(
            z=matrix.values,
            x=matrix.columns.tolist(),
            y=matrix.columns.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=np.round(matrix.values, settings.corr_round_precision),
            texttemplate="%{text}",
            colorbar={"title": "r"},
        )
    )
    fig.update_layout(title=title)
    return apply_dscompanion_theme(fig)
