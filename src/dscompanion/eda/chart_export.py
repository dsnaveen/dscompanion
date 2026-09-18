"""Save EDAReport's charts as standalone matplotlib image files.

Deliberately matplotlib/seaborn-based, not Plotly — Plotly's static image
export (``fig.to_image()``/``write_image()``) requires the optional
``kaleido`` package for every format, including SVG (the old kaleido-free
``orca`` engine was removed years ago). matplotlib and seaborn are already
hard dscompanion dependencies with built-in, dependency-free PNG/SVG
backends, so this avoids adding any new dependency. Mirrors the same
reasoning already documented in
``dscompanion.docs.model_card.ModelCard._write_interactions_excel_sheet``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from dscompanion.config import settings
from dscompanion.utils.metrics import woe_bins

if TYPE_CHECKING:
    from dscompanion.eda.report import EDAReport

logger = logging.getLogger(__name__)

__all__ = ["export_charts"]

_FILENAME_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_filename(name: str) -> str:
    """Replace filesystem-unsafe characters (spaces, slashes, etc.) with underscores."""
    return _FILENAME_UNSAFE.sub("_", name).strip("_") or "col"


def _save(fig: Any, path: Path, format: str, dpi: int) -> Path:
    out_path = path.with_suffix(f".{format}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format=format, dpi=dpi, bbox_inches="tight")
    return out_path


def _select_categorical_columns(train_X: pd.DataFrame, max_cols: int) -> list[str]:
    """Pick up to ``max_cols`` non-numeric columns, lowest-cardinality first.

    Lower cardinality is preferred because it produces more readable
    stacked-bar/heatmap/box-plot charts — mirrors ``interaction_columns()``'s
    own IV-ranked capping for numeric columns, just with a different ranking
    criterion (no target-based ranking exists for arbitrary categorical
    pairs).
    """
    cat_cols = [c for c in train_X.columns if not pd.api.types.is_numeric_dtype(train_X[c])]
    return sorted(cat_cols, key=lambda c: train_X[c].nunique())[:max_cols]


def _cap_cardinality(s: pd.Series, top_n: int, other: str = "Other") -> pd.Series:
    """Bucket every value outside the top-N most frequent into ``other``."""
    keep = s.value_counts().nlargest(top_n).index
    return s.where(s.isin(keep), other)


def _sort_by_marginal(ct: pd.DataFrame) -> pd.DataFrame:
    """Reorder a contingency table's rows/columns by descending marginal frequency.

    Alphabetical ordering hides structure; sorting by row/column totals
    surfaces the dominant categories first.
    """
    return ct.loc[
        ct.sum(axis=1).sort_values(ascending=False).index,
        ct.sum(axis=0).sort_values(ascending=False).index,
    ]


def _cramers_v_and_residuals(ct: pd.DataFrame) -> tuple[float, pd.DataFrame, bool]:
    """Compute Cramér's V (effect size) and standardized Pearson residuals for a contingency table.

    Returns:
        tuple[float, pd.DataFrame, bool]: ``(cramers_v, residuals, reliable)``.
        ``reliable`` is ``False`` when more than 20% of the table's expected
        cell counts fall below 5 — the chi-square approximation (and
        anything derived from it, including Cramér's V) is unreliable in
        that regime; callers should flag this rather than presenting a
        precise-looking but misleading number.
    """
    chi2, _p, _dof, expected = chi2_contingency(ct)
    residuals = (ct - expected) / np.sqrt(expected)
    n = ct.values.sum()
    min_dim = min(ct.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 else 0.0
    reliable = bool((expected < 5).mean() <= 0.2)
    return cramers_v, residuals, reliable


def _add_trend_line(ax: Any, x: pd.Series, y: pd.Series, method: str, poly_degree: int) -> None:
    """Overlay a fit line on an existing scatter/hexbin Axes.

    A no-op for ``method="none"`` or fewer than 3 non-null point pairs. Never
    raises -- a failed fit (e.g. degenerate input) is logged at debug and
    skipped, since the underlying chart must still be saved either way.

    Args:
        ax (Any): The matplotlib Axes to draw the trend line onto.
        x (pd.Series): The x-axis values already plotted on ``ax``.
        y (pd.Series): The y-axis values already plotted on ``ax``.
        method (str): One of ``"none"``, ``"linear"``, ``"polynomial"``,
            ``"loess"``.
        poly_degree (int): Polynomial degree used only when
            ``method="polynomial"``.

    Returns:
        None: Draws directly onto ``ax`` as a side effect.
    """
    if method == "none":
        return
    paired = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(paired) < 3:
        logger.debug("Skipping trend line -- fewer than 3 non-null point pairs")
        return
    paired = paired.sort_values("x")
    try:
        if method == "linear":
            coeffs = np.polyfit(paired["x"], paired["y"], 1)
            x_line = np.linspace(paired["x"].min(), paired["x"].max(), 100)
            ax.plot(x_line, np.polyval(coeffs, x_line), color="firebrick", linewidth=2)
        elif method == "polynomial":
            coeffs = np.polyfit(paired["x"], paired["y"], poly_degree)
            x_line = np.linspace(paired["x"].min(), paired["x"].max(), 100)
            ax.plot(x_line, np.polyval(coeffs, x_line), color="firebrick", linewidth=2)
        elif method == "loess":
            from statsmodels.nonparametric.smoothers_lowess import lowess

            smoothed = lowess(paired["y"], paired["x"])
            ax.plot(smoothed[:, 0], smoothed[:, 1], color="firebrick", linewidth=2)
    except Exception as exc:
        logger.debug("Trend line fit failed (non-fatal): %s", exc)


_AUTO_STYLE_MAX_CATEGORIES_FOR_SUMMARY = 15
_AUTO_STYLE_MAX_CATEGORIES_FOR_DETAIL = 4
_AUTO_STYLE_MAX_N_FOR_STRIP = 30
_AUTO_STYLE_MIN_N_FOR_DENSITY = 20


def _resolve_numeric_categorical_style(style: str, n_categories: int, avg_n: float) -> str:
    """Pick a concrete numeric×categorical chart style when ``style="auto"``.

    Mirrors the cardinality-adaptive philosophy already used for
    categorical×categorical pairs: one chart type per pair, chosen from data
    shape, rather than every possible chart type per pair (issue #8).

    Args:
        style (str): The requested style -- one of ``"auto"``, ``"box"``,
            ``"violin"``, ``"strip"``, ``"mean_errorbar"``, ``"kde"``. Any
            value other than ``"auto"`` is returned unchanged.
        n_categories (int): Number of distinct categories in the grouping
            column.
        avg_n (float): Average number of rows per category
            (``len(train_X) / n_categories``).

    Returns:
        str: The resolved style -- always one of ``"box"``, ``"violin"``,
        ``"strip"``, ``"mean_errorbar"``, ``"kde"``.
    """
    if style != "auto":
        return style
    if n_categories > _AUTO_STYLE_MAX_CATEGORIES_FOR_SUMMARY:
        return "mean_errorbar"
    if (
        n_categories <= _AUTO_STYLE_MAX_CATEGORIES_FOR_DETAIL
        and avg_n <= _AUTO_STYLE_MAX_N_FOR_STRIP
    ):
        return "strip"
    if avg_n < _AUTO_STYLE_MIN_N_FOR_DENSITY:
        return "box"
    if n_categories <= _AUTO_STYLE_MAX_CATEGORIES_FOR_DETAIL:
        return "kde"
    return "violin"


def export_charts(
    report: "EDAReport",
    output_dir: str | Path,
    format: str = "png",
    dpi: int = 0,
    bivariate_top_n: int = 0,
    bivariate_clip_lower_pct: float = 0.05,
    bivariate_clip_upper_pct: float = 0.05,
    numeric_categorical_style: str = "auto",
    numeric_interaction_trend_line: str = "loess",
    numeric_interaction_trend_poly_degree: int = 2,
    use_target_hue: bool = False,
) -> dict[str, Path]:
    """Render every EDAReport chart category to standalone image files under ``output_dir``.

    Builds its own matplotlib/seaborn figures from ``EDAReport``'s existing
    data-accessor methods (not by converting the Plotly figures those same
    methods already return elsewhere) — see the module docstring for why.

    Args:
        report (EDAReport): A fitted report (``run_all()`` already called).
        output_dir (str | Path): Root directory for the exported charts.
            Created if it doesn't exist. Populated with subdirectories:
            ``univariate/{numeric,categorical}/``, ``bivariate/``,
            ``multivariate/`` (correlation heatmap when ≥2 numeric columns
            exist; ``multivariate/interactions/`` with numeric×numeric
            scatter pairs, categorical×categorical contingency charts, and
            numeric×categorical box plots — each only when the relevant
            column types exist), ``missingness/``.
        format (str): ``"png"`` (default) or ``"svg"``. Both are native
            matplotlib backends — no optional dependency either way.
        dpi (int): Pixel density for PNG output (ignored for SVG, which is
            resolution-independent). ``0`` uses
            ``settings.eda_chart_export_dpi``. Defaults to ``0``.
        bivariate_top_n (int): Max number of features to export bivariate
            (target-rate-by-bin) charts for, ranked by Information Value.
            ``0`` uses ``settings.eda_bivariate_chart_top_n``. Defaults to
            ``0``.
        bivariate_clip_lower_pct (float): Lower-tail fraction clipped from
            a numeric feature before quantile-binning it for its bivariate
            chart (and before numeric×numeric interaction scatter's
            "clipped" variant), so a few extreme values don't distort the
            chart. Has no effect on categorical features. Defaults to
            ``0.05``.
        bivariate_clip_upper_pct (float): Upper-tail fraction clipped
            before the same binning/scatter. Defaults to ``0.05``.
        numeric_categorical_style (str): Chart style for numeric×categorical
            interaction pairs -- one of ``"auto"``, ``"box"``, ``"violin"``,
            ``"strip"``, ``"mean_errorbar"``, ``"kde"``. ``"auto"`` (default)
            adaptively resolves a single style per pair based on category
            count and average rows per category -- see
            ``_resolve_numeric_categorical_style``.
        numeric_interaction_trend_line (str): Trend-line overlay for the
            ``_raw``/``_clipped`` numeric×numeric scatter/hexbin charts --
            one of ``"none"``, ``"linear"``, ``"loess"``, ``"polynomial"``.
            Not applied to the ``_logscale`` variant. Defaults to
            ``"loess"``.
        numeric_interaction_trend_poly_degree (int): Polynomial degree used
            only when ``numeric_interaction_trend_line="polynomial"``.
            Defaults to ``2``.
        use_target_hue (bool): Color numeric×numeric scatter points
            (raw/clipped variants, not hexbin or log-scale) and
            numeric×categorical box/violin/strip charts by the target class.
            Only applied when the target is binary
            (``train_y.nunique() == 2``) -- silently ignored otherwise.
            Defaults to ``False``.

    Returns:
        dict[str, Path]: Maps a category-relative key (e.g.
        ``"univariate/numeric/age"``, ``"multivariate/correlation_heatmap"``)
        to the absolute path written. Every value in the dict exists on
        disk. Categorical×categorical pairs additionally get a
        ``"<key>_table"`` entry pointing at the persisted contingency table
        CSV (the reusable artifact — charts get regenerated).

    Raises:
        RuntimeError: If ``report.run_all()`` has not been called yet
            (propagated from ``EDAReport``'s own accessor methods).
        ValueError: If ``format`` is not ``"png"`` or ``"svg"``.
    """
    if format not in ("png", "svg"):
        raise ValueError(f"format must be 'png' or 'svg', got {format!r}")
    valid_styles = {"auto", "box", "violin", "strip", "mean_errorbar", "kde"}
    if numeric_categorical_style not in valid_styles:
        raise ValueError(
            f"numeric_categorical_style must be one of {valid_styles}, "
            f"got {numeric_categorical_style!r}"
        )
    dpi = dpi or settings.eda_chart_export_dpi
    bivariate_top_n = bivariate_top_n or settings.eda_bivariate_chart_top_n

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    report._check_run()
    output_dir = Path(output_dir)
    train_X = report._split.train_X
    paths: dict[str, Path] = {}

    # ── Univariate ────────────────────────────────────────────────────────
    for col in train_X.columns:
        series = train_X[col]
        safe = _safe_filename(col)
        fig, ax = plt.subplots(figsize=(6, 4))
        try:
            if isinstance(series.dtype, pd.CategoricalDtype) or not pd.api.types.is_numeric_dtype(
                series
            ):
                key = f"univariate/categorical/{safe}"
                counts = report.categorical_value_counts(
                    col, top_n=settings.eda_top_categorical_values
                )
                ax.bar(counts.index.astype(str), counts.values, color="#2a78d6")
                ax.set_title(f"{col} — value counts")
                ax.tick_params(axis="x", rotation=45)
            else:
                key = f"univariate/numeric/{safe}"
                clean = report.numeric_clean_series(col)
                ax.hist(clean, bins=settings.eda_histogram_bins, color="#2a78d6")
                ax.set_title(f"{col} — distribution")
            fig.tight_layout()
            paths[key] = _save(fig, output_dir / key, format, dpi)
        except Exception as exc:
            logger.warning("Skipping univariate chart for column %r: %s", col, exc)
        finally:
            plt.close(fig)

    # ── Bivariate (top-N by IV) ──────────────────────────────────────────
    iv_df = report.iv_table()
    top_features = iv_df["feature"].head(bivariate_top_n).tolist() if len(iv_df) else []
    for feature in top_features:
        safe = _safe_filename(feature)
        key = f"bivariate/{safe}"
        fig, ax = plt.subplots(figsize=(6, 4))
        try:
            raw_series = train_X[feature]
            if pd.api.types.is_numeric_dtype(raw_series) and (
                bivariate_clip_lower_pct > 0 or bivariate_clip_upper_pct > 0
            ):
                clean = raw_series.dropna()
                lower_bound = clean.quantile(bivariate_clip_lower_pct)
                upper_bound = clean.quantile(1 - bivariate_clip_upper_pct)
                mask = raw_series.between(lower_bound, upper_bound)
                clipped_series = raw_series[mask]
                clipped_target = report._split.train_y.loc[clipped_series.index]
                raw_bins = woe_bins(clipped_series, clipped_target, settings.default_iv_bins)
                bin_df = raw_bins.rename(columns={"bin": "bin_label", "event_rate": "target_rate"})
            else:
                bin_df = report.target_rate_by_bin(feature)
            ax.bar(bin_df["bin_label"].astype(str), bin_df["target_rate"], color="#eb6834")
            ax.set_title(f"{feature} — target rate by bin")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            paths[key] = _save(fig, output_dir / key, format, dpi)
        except Exception as exc:
            logger.warning("Skipping bivariate chart for feature %r: %s", feature, exc)
        finally:
            plt.close(fig)

    # ── Multivariate: correlation heatmap ────────────────────────────────
    corr_matrix = report.correlation_matrix()
    if corr_matrix.shape[0] >= 2:
        key = "multivariate/correlation_heatmap"
        fig, ax = plt.subplots(figsize=(7, 6))
        try:
            sns.heatmap(corr_matrix, cmap="RdBu", center=0, annot=False, ax=ax)
            ax.set_title("Correlation heatmap")
            fig.tight_layout()
            paths[key] = _save(fig, output_dir / key, format, dpi)
        except Exception as exc:
            logger.warning("Skipping correlation heatmap: %s", exc)
        finally:
            plt.close(fig)

    # Column selection for the pairwise interaction charts below is independent
    # of the correlation-heatmap gate above (a dataset with <2 numeric columns
    # can still have categorical pairs worth charting).
    numeric_cols = [c for c in report.interaction_columns() if c in train_X.columns]
    categorical_cols = _select_categorical_columns(train_X, max_cols=len(numeric_cols) or 8)

    # Computed once and reused by both the numeric x numeric and numeric x
    # categorical sections below -- detected data-driven (nunique() == 2)
    # rather than via an explicit "task" parameter, since chart_export.py is
    # otherwise fully task-agnostic.
    target_hue: pd.Series | None = None
    if use_target_hue:
        train_y = report._split.train_y
        if train_y.nunique() == 2:
            target_hue = train_y

    # ── Numeric x Numeric: scatter, 3 variants (raw / clipped / log-scale) ──
    if len(numeric_cols) >= 2:
        data = train_X[numeric_cols]
        if len(data) > settings.max_eda_rows:
            data = data.sample(n=settings.max_eda_rows, random_state=settings.random_state)
        use_hexbin = len(data) > settings.eda_interaction_hexbin_row_threshold

        def _scatter_or_hexbin(fig, ax, x, y):
            if use_hexbin:
                hb = ax.hexbin(x, y, gridsize=30, cmap="Blues")
                fig.colorbar(hb, ax=ax, label="count")
            elif target_hue is not None:
                sns.scatterplot(x=x, y=y, hue=target_hue.loc[x.index], ax=ax, s=12, alpha=0.5)
            else:
                sns.scatterplot(x=x, y=y, ax=ax, s=12, alpha=0.5)

        for i, c1 in enumerate(numeric_cols):
            for c2 in numeric_cols[i + 1 :]:
                base_key = f"multivariate/interactions/{_safe_filename(c1)}_vs_{_safe_filename(c2)}"
                x_raw, y_raw = data[c1], data[c2]

                # 1. Without clipping
                key = f"{base_key}_raw"
                fig, ax = plt.subplots(figsize=(5, 3.5))
                try:
                    _scatter_or_hexbin(fig, ax, x_raw, y_raw)
                    _add_trend_line(
                        ax,
                        x_raw,
                        y_raw,
                        numeric_interaction_trend_line,
                        numeric_interaction_trend_poly_degree,
                    )
                    ax.set_title(f"{c1} vs {c2} — raw")
                    ax.set_xlabel(c1)
                    ax.set_ylabel(c2)
                    fig.tight_layout()
                    paths[key] = _save(fig, output_dir / key, format, dpi)
                except Exception as exc:
                    logger.warning("Skipping raw interaction %s vs %s: %s", c1, c2, exc)
                finally:
                    plt.close(fig)

                # 2. With clipping
                key = f"{base_key}_clipped"
                fig, ax = plt.subplots(figsize=(5, 3.5))
                try:
                    x_clip = x_raw.clip(
                        x_raw.quantile(bivariate_clip_lower_pct),
                        x_raw.quantile(1 - bivariate_clip_upper_pct),
                    )
                    y_clip = y_raw.clip(
                        y_raw.quantile(bivariate_clip_lower_pct),
                        y_raw.quantile(1 - bivariate_clip_upper_pct),
                    )
                    _scatter_or_hexbin(fig, ax, x_clip, y_clip)
                    _add_trend_line(
                        ax,
                        x_clip,
                        y_clip,
                        numeric_interaction_trend_line,
                        numeric_interaction_trend_poly_degree,
                    )
                    ax.set_title(f"{c1} vs {c2} — clipped")
                    ax.set_xlabel(c1)
                    ax.set_ylabel(c2)
                    fig.tight_layout()
                    paths[key] = _save(fig, output_dir / key, format, dpi)
                except Exception as exc:
                    logger.warning("Skipping clipped interaction %s vs %s: %s", c1, c2, exc)
                finally:
                    plt.close(fig)

                # 3. Log scale (only when both columns are strictly positive)
                if (x_raw > 0).all() and (y_raw > 0).all():
                    key = f"{base_key}_logscale"
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    try:
                        _scatter_or_hexbin(fig, ax, x_raw, y_raw)
                        ax.set_xscale("log")
                        ax.set_yscale("log")
                        ax.set_title(f"{c1} vs {c2} — log scale")
                        ax.set_xlabel(c1)
                        ax.set_ylabel(c2)
                        fig.tight_layout()
                        paths[key] = _save(fig, output_dir / key, format, dpi)
                    except Exception as exc:
                        logger.warning("Skipping log-scale interaction %s vs %s: %s", c1, c2, exc)
                    finally:
                        plt.close(fig)
                else:
                    logger.debug(
                        "Skipping log-scale interaction %s vs %s -- non-positive values present",
                        c1,
                        c2,
                    )

    # ── Categorical x Categorical: cardinality-adaptive ──────────────────
    # Per notes/bivariate_categorical_plots.md: <=5x5 levels -> 100% stacked
    # bar (proportions, robust to imbalanced group sizes); otherwise ->
    # standardized-residual heatmap, sorted by marginal frequency, with a
    # >30-level long tail bucketed into "Other" before the crosstab.
    for i, c1 in enumerate(categorical_cols):
        for c2 in categorical_cols[i + 1 :]:
            key = f"multivariate/interactions/{_safe_filename(c1)}_vs_{_safe_filename(c2)}"
            fig, ax = plt.subplots(figsize=(7, 5))
            try:
                s1 = train_X[c1].astype(str)
                s2 = train_X[c2].astype(str)
                if s1.nunique() > 30:
                    s1 = _cap_cardinality(s1, top_n=29)
                if s2.nunique() > 30:
                    s2 = _cap_cardinality(s2, top_n=29)
                ct = _sort_by_marginal(pd.crosstab(s1, s2))
                cramers_v, residuals, reliable = _cramers_v_and_residuals(ct)
                v_label = f"{cramers_v:.3f}" + ("" if reliable else " (sparse, unreliable)")

                if ct.shape[0] <= 5 and ct.shape[1] <= 5:
                    # pandas plots ct's index (c1) along the x-axis, one bar group per
                    # c1 value, with c2's values as the stacked/legend segments.
                    proportions = ct.div(ct.sum(axis=1), axis=0)
                    proportions.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
                    ax.set_title(f"{c1} vs {c2} — 100% stacked (Cramér's V={v_label})")
                    ax.set_xlabel(c1)
                    ax.set_ylabel("proportion")
                    ax.legend(title=c2, bbox_to_anchor=(1.02, 1), loc="upper left")
                else:
                    # seaborn plots ct's index (c1) on the y-axis, columns (c2) on the x-axis.
                    sns.heatmap(
                        residuals,
                        center=0,
                        cmap="RdBu_r",
                        annot=(residuals.size <= 100),
                        fmt=".1f",
                        ax=ax,
                    )
                    ax.set_title(f"{c1} vs {c2} — std. residuals (Cramér's V={v_label})")
                    ax.set_xlabel(c2)
                    ax.set_ylabel(c1)
                ax.tick_params(axis="x", rotation=45)
                fig.tight_layout()
                paths[key] = _save(fig, output_dir / key, format, dpi)

                # Persist the contingency table itself, not just the rendered
                # image -- per notes/bivariate_categorical_plots.md's
                # production notes, the table is the reusable artifact.
                csv_path = (output_dir / key).with_suffix(".csv")
                ct.to_csv(csv_path)
                paths[f"{key}_table"] = csv_path
            except Exception as exc:
                logger.warning("Skipping categorical pair %s vs %s: %s", c1, c2, exc)
            finally:
                plt.close(fig)

    # ── Numeric x Categorical: single, cardinality-adaptive chart per pair ──
    # One chart per pair regardless of style, to avoid multiplying issue #8's
    # already-flagged runtime cost.
    for num_col in numeric_cols:
        for cat_col in categorical_cols:
            key = (
                f"multivariate/interactions/{_safe_filename(num_col)}_by_{_safe_filename(cat_col)}"
            )
            fig, ax = plt.subplots(figsize=(7, 5))
            try:
                cat_series = train_X[cat_col].astype(str)
                num_series = train_X[num_col]
                n_categories = cat_series.nunique()
                avg_n = len(train_X) / max(n_categories, 1)
                style = _resolve_numeric_categorical_style(
                    numeric_categorical_style, n_categories, avg_n
                )
                hue_kwargs = {"hue": target_hue, "legend": True} if target_hue is not None else {}

                if style == "box":
                    sns.boxplot(x=cat_series, y=num_series, ax=ax, **hue_kwargs)
                elif style == "violin":
                    sns.violinplot(x=cat_series, y=num_series, ax=ax, **hue_kwargs)
                elif style == "strip":
                    sns.stripplot(
                        x=cat_series, y=num_series, ax=ax, alpha=0.6, jitter=True, **hue_kwargs
                    )
                elif style == "mean_errorbar":
                    grouped = pd.DataFrame({"cat": cat_series, "num": num_series}).groupby("cat")[
                        "num"
                    ]
                    stats = grouped.agg(["mean", "sem"]).dropna()
                    ax.errorbar(stats.index, stats["mean"], yerr=stats["sem"], fmt="o", capsize=4)
                elif style == "kde":
                    for level, sub in pd.DataFrame({"cat": cat_series, "num": num_series}).groupby(
                        "cat"
                    )["num"]:
                        sns.kdeplot(sub.dropna(), ax=ax, label=str(level), fill=True, alpha=0.4)
                    ax.legend(title=cat_col)

                ax.set_title(f"{num_col} by {cat_col} ({style})")
                ax.set_xlabel(cat_col if style != "kde" else num_col)
                ax.set_ylabel(num_col if style != "kde" else "density")
                ax.tick_params(axis="x", rotation=45)
                fig.tight_layout()
                paths[key] = _save(fig, output_dir / key, format, dpi)
            except Exception as exc:
                logger.warning(
                    "Skipping numeric-by-categorical chart %s by %s: %s", num_col, cat_col, exc
                )
            finally:
                plt.close(fig)

    # ── Missingness ───────────────────────────────────────────────────────
    key = "missingness/missing_values"
    fig, ax = plt.subplots(figsize=(7, 4))
    try:
        num_df = report.numeric_summary()[["feature", "missing_pct"]]
        cat_df = report.categorical_summary()[["feature", "missing_pct"]]
        combined = pd.concat([num_df, cat_df], ignore_index=True).sort_values(
            "missing_pct", ascending=False
        )
        ax.bar(combined["feature"], combined["missing_pct"] * 100, color="#c0392b")
        ax.set_title("Missing values by feature")
        ax.set_ylabel("missing %")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        paths[key] = _save(fig, output_dir / key, format, dpi)
    except Exception as exc:
        logger.warning("Skipping missing-values chart: %s", exc)
    finally:
        plt.close(fig)

    key = "missingness/missing_matrix"
    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        matrix = report.missing_matrix()
        if matrix.shape[1] > 0:
            sns.heatmap(matrix, cmap=["#f9f9f7", "#c0392b"], cbar=False, ax=ax)
            ax.set_title("Missingness matrix")
            fig.tight_layout()
            paths[key] = _save(fig, output_dir / key, format, dpi)
    except Exception as exc:
        logger.warning("Skipping missingness matrix: %s", exc)
    finally:
        plt.close(fig)

    key = "missingness/missing_correlation"
    fig, ax = plt.subplots(figsize=(7, 6))
    try:
        missing_corr = report.missing_correlation()
        if missing_corr.shape[0] >= 2:
            sns.heatmap(missing_corr, cmap="RdBu", center=0, annot=False, ax=ax)
            ax.set_title("Missingness correlation")
            fig.tight_layout()
            paths[key] = _save(fig, output_dir / key, format, dpi)
    except Exception as exc:
        logger.warning("Skipping missingness correlation: %s", exc)
    finally:
        plt.close(fig)

    return paths
