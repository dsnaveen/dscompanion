"""Glossary — static reference page explaining every Step 5 feature-processing
transformation: what it does, its formula, and a live-computed before/after
example with edge cases.

Every example is computed live, at render time, by calling the exact same
``TRANSFORMER_REGISTRY`` factory and ``.fit_transform()`` that Step 5's real
recipe chain uses — never a hand-written static table. This guarantees the
numbers shown always match production behaviour, including if the underlying
transform logic ever changes (important for a bank's audit-defensible
tooling). See ``plans/glossary-page.md`` for the design this module implements.

Fully self-contained — no dependency on any uploaded dataset or session state
from Interactive/Full Pipeline mode.

Local-dev Streamlit UI only — never imported by dscompanion's runtime code.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from typing import Literal

import pandas as pd
import streamlit as st

from dscompanion.features.registry import TRANSFORMER_REGISTRY

logger = logging.getLogger(__name__)

__all__ = ["render_glossary"]

_FormulaKind = Literal["latex", "code", "text"]


@dataclasses.dataclass(frozen=True)
class GlossaryEntry:
    """One Step 5 transformation's full glossary content.

    Args:
        name (str): Registry key — must match a ``TRANSFORMER_REGISTRY``
            entry exactly (enforced by a test that loops over the registry).
        category (str): One of the 8 navigation categories in ``_CATEGORIES``.
        what_it_does (str): Plain-language, non-expert-friendly explanation.
        formula_kind ("latex" | "code" | "text"): How ``formula`` should be
            rendered — ``st.latex`` for clean closed-form math, ``st.code``
            for pseudocode/algorithm sketches, ``st.markdown`` for prose.
        formula (str): The formula or algorithm description itself.
        build_demo (Callable[[], tuple[pd.DataFrame, pd.Series | None]]):
            Returns a small illustrative ``(X, y)`` pair — ``y`` is ``None``
            for transforms that don't need a target.
        edge_cases (list[str]): Plain-language bullets — NaN handling, unseen
            categories, degenerate columns, etc.
        demo_params (dict): Factory-param overrides applied only for this
            page's demo (e.g. a lowered cardinality threshold so a tiny toy
            column actually triggers processing) — never changes the real
            pipeline default.
    """

    name: str
    category: str
    what_it_does: str
    formula_kind: _FormulaKind
    formula: str
    build_demo: Callable[[], tuple[pd.DataFrame, pd.Series | None]]
    edge_cases: list[str]
    demo_params: dict = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# Demo datasets — small, banking-flavored, shared across entries where the
# same shape/values usefully illustrate more than one transform.
# ---------------------------------------------------------------------------


def _demo_balance_positive() -> tuple[pd.DataFrame, None]:
    return pd.DataFrame({"balance": [500, 2_000, 8_000, 15_000, 45_000, 120_000, 300_000]}), None


def _demo_cashflow_with_zero_negative() -> tuple[pd.DataFrame, None]:
    return pd.DataFrame({"net_cashflow": [-5_000, -800, 0, 250, 1_500, 8_000, 25_000]}), None


def _demo_pct_change_bounded() -> tuple[pd.DataFrame, None]:
    # log1p requires every value > -1 — unlike net_cashflow above (which can go
    # arbitrarily negative and needs yeo_johnson), a percentage-change column is
    # naturally bounded below by -1 (a 100% decrease), making it a valid log1p input.
    return pd.DataFrame({"pct_change_in_balance": [-0.6, -0.2, 0.0, 0.15, 0.5, 1.2, 3.5]}), None


def _demo_income_with_outlier() -> tuple[pd.DataFrame, None]:
    return (
        pd.DataFrame({"income": [28_000, 31_000, 33_000, 35_000, 38_000, 42_000, 500_000]}),
        None,
    )


def _demo_age() -> tuple[pd.DataFrame, None]:
    return pd.DataFrame({"age": [22, 25, 29, 33, 38, 45, 52, 61, 68, 75]}), None


def _demo_age_with_target() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame({"age": [22, 25, 29, 33, 38, 45, 52, 61, 68, 75]})
    y = pd.Series([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], name="default_flag")
    return X, y


def _demo_segment() -> tuple[pd.DataFrame, None]:
    seg = ["retail", "retail", "retail", "premium", "premium", "mass_affluent"]
    return pd.DataFrame({"segment": seg}), None


def _demo_segment_with_target() -> tuple[pd.DataFrame, pd.Series]:
    seg = [
        "retail",
        "retail",
        "retail",
        "premium",
        "premium",
        "mass_affluent",
        "mass_affluent",
        "retail",
    ]
    X = pd.DataFrame({"segment": seg})
    y = pd.Series([1, 1, 0, 0, 0, 1, 0, 1], name="default_flag")
    return X, y


def _demo_segment_with_rare() -> tuple[pd.DataFrame, None]:
    seg = ["retail", "retail", "retail", "retail", "premium", "premium", "niche"]
    return pd.DataFrame({"segment": seg}), None


def _demo_high_cardinality_with_target() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame({"branch_code": ["B01", "B02", "B03", "B01", "B04", "B02", "B05", "B01"]})
    y = pd.Series([1, 0, 1, 1, 0, 0, 1, 1], name="default_flag")
    return X, y


def _demo_balance_with_missing() -> tuple[pd.DataFrame, None]:
    vals = [500.0, None, 15_000.0, None, 45_000.0, 120_000.0, 8_000.0]
    return pd.DataFrame({"balance": vals}), None


def _demo_age_income() -> tuple[pd.DataFrame, None]:
    X = pd.DataFrame(
        {
            "age": [22, 29, 35, 41, 48, 55, 62, 70],
            "income": [28_000, 32_000, 38_000, 45_000, 52_000, 60_000, 65_000, 70_000],
        }
    )
    return X, None


# ---------------------------------------------------------------------------
# Navigation categories, in display order.
# ---------------------------------------------------------------------------

_CATEGORIES: list[str] = [
    "Missing values",
    "Outlier handling",
    "Distribution shape",
    "Binning",
    "Ranking & scaling",
    "Categorical encoding",
    "Feature expansion",
    "Data augmentation",
]

# ---------------------------------------------------------------------------
# The 25 entries — one per TRANSFORMER_REGISTRY key.
# ---------------------------------------------------------------------------

_ENTRIES: list[GlossaryEntry] = [
    GlossaryEntry(
        name="impute",
        category="Missing values",
        what_it_does=(
            "Fills missing values column-by-column. For numeric columns it picks the median "
            "or the mean automatically, based on how skewed the column is — median for "
            "skewed data (like balances), mean for roughly symmetric data. Categorical "
            "columns are filled with the most common value."
        ),
        formula_kind="text",
        formula=(
            "Numeric: median(x) if |skewness| ≥ threshold, else mean(x). "
            'Categorical: mode(x), or "MISSING" if no mode exists.'
        ),
        build_demo=_demo_balance_with_missing,
        edge_cases=[
            "A column that is entirely missing has nothing to learn a fill value from — it is "
            "skipped and stays all-missing.",
            "Optionally adds a `{column}_was_missing` 0/1 indicator column when the missing "
            "rate is high enough to be informative on its own.",
        ],
    ),
    GlossaryEntry(
        name="clip_lower",
        category="Outlier handling",
        what_it_does=(
            "Caps unusually small values at a training-data percentile, so a handful of "
            "extreme low outliers can't dominate a model."
        ),
        formula_kind="latex",
        formula=r"x' = \max\bigl(x,\ \mathrm{quantile}_{lower}(x_{train})\bigr)",
        build_demo=_demo_income_with_outlier,
        demo_params={"lower": 0.2},
        edge_cases=[
            "Only the lower tail moves — large values pass through unchanged.",
            "The cap is learned once from training data and reused as-is at scoring time, "
            "even if new data has more extreme lows.",
            "A column that is entirely missing has no percentile to learn and is skipped.",
        ],
    ),
    GlossaryEntry(
        name="clip_upper",
        category="Outlier handling",
        what_it_does=(
            "Caps unusually large values at a training-data percentile — the mirror image of "
            "clip_lower, for outliers on the high end (like one customer with a huge balance)."
        ),
        formula_kind="latex",
        formula=r"x' = \min\bigl(x,\ \mathrm{quantile}_{1-upper}(x_{train})\bigr)",
        build_demo=_demo_income_with_outlier,
        demo_params={"upper": 0.2},
        edge_cases=[
            "Only the upper tail moves — small values pass through unchanged.",
            "The cap is learned once from training data and reused as-is at scoring time.",
        ],
    ),
    GlossaryEntry(
        name="clip_both",
        category="Outlier handling",
        what_it_does=(
            "Caps both unusually small and unusually large values at training-data percentiles."
        ),
        formula_kind="latex",
        formula=(
            r"x' = \mathrm{clip}\bigl(x,\ \mathrm{quantile}_{lower}(x_{train}),"
            r"\ \mathrm{quantile}_{1-upper}(x_{train})\bigr)"
        ),
        build_demo=_demo_income_with_outlier,
        demo_params={"lower": 0.2, "upper": 0.2},
        edge_cases=[
            "Both tails move — this is the combination of clip_lower and clip_upper in one step.",
            "The default clip is a modest 1% each side; this demo uses a wider 20% clip so the "
            "effect is visible on a tiny example.",
        ],
    ),
    GlossaryEntry(
        name="clip_iqr",
        category="Outlier handling",
        what_it_does=(
            "Caps values outside a multiple of the interquartile range (IQR) — the classic "
            "'box plot fence' rule. Unlike clip_lower/upper/both, the cap isn't a fixed "
            "percentile; it's derived from the column's own spread, so it adapts to how tight "
            "or wide the middle 50% of the data actually is."
        ),
        formula_kind="latex",
        formula=(
            r"x' = \mathrm{clip}\bigl(x,\ Q_1 - k \cdot \mathrm{IQR},\ Q_3 + k \cdot \mathrm{IQR}"
            r"\bigr),\quad \mathrm{IQR} = Q_3 - Q_1"
        ),
        build_demo=_demo_income_with_outlier,
        edge_cases=[
            "If the middle 50% of training values are all identical (IQR = 0), there's no "
            "meaningful fence to compute and the column is skipped rather than collapsed to a "
            "single value.",
            "The fence is learned once from training data and reused as-is at scoring time.",
        ],
    ),
    GlossaryEntry(
        name="clip_zscore",
        category="Outlier handling",
        what_it_does=(
            "Caps values more than k standard deviations from the training mean — the "
            "classic rule for roughly bell-curve-shaped data. Because the mean/std are computed "
            "on the same data being capped, a single huge outlier can itself widen the fence "
            "that's meant to catch it."
        ),
        formula_kind="latex",
        formula=r"x' = \mathrm{clip}\bigl(x,\ \bar{x} - k\sigma,\ \bar{x} + k\sigma\bigr)",
        build_demo=_demo_income_with_outlier,
        edge_cases=[
            "A column with zero variance (every training value identical) has no std to scale "
            "by and is skipped.",
            "Less robust to extreme outliers than clip_iqr or clip_mad, since the very values "
            "being capped inflate the mean/std used to set the cap.",
        ],
    ),
    GlossaryEntry(
        name="clip_mad",
        category="Outlier handling",
        what_it_does=(
            "Caps values more than k scaled median-absolute-deviations from the training "
            "median — the most outlier-resistant of the three statistical capping rules, since "
            "median and MAD are barely moved by extreme values (unlike mean/std)."
        ),
        formula_kind="latex",
        formula=(
            r"x' = \mathrm{clip}\bigl(x,\ m - k \cdot 1.4826\,\mathrm{MAD},"
            r"\ m + k \cdot 1.4826\,\mathrm{MAD}\bigr)"
        ),
        build_demo=_demo_income_with_outlier,
        edge_cases=[
            "A column where at least half the training values are identical (MAD = 0) has no "
            "meaningful fence and is skipped.",
            "The 1.4826 constant makes MAD comparable to a normal distribution's standard "
            "deviation — a standard statistical convention, not a tunable choice.",
        ],
    ),
    GlossaryEntry(
        name="log",
        category="Distribution shape",
        what_it_does=(
            "Compresses right-skewed positive values (like income or balance) by taking the "
            "natural log, pulling in extreme values so the column looks closer to a normal "
            "distribution."
        ),
        formula_kind="latex",
        formula=r"x' = \ln(x)",
        build_demo=_demo_balance_positive,
        edge_cases=[
            "Every value must be strictly greater than 0 — a column with 0 or negative values "
            "will fail to fit. Use log1p or yeo_johnson instead.",
            "A value outside the range seen during fit (e.g. a new non-positive value at "
            "scoring time) becomes a blank/missing value rather than an error.",
        ],
    ),
    GlossaryEntry(
        name="log1p",
        category="Distribution shape",
        what_it_does=(
            "Like log, but works on columns that can be exactly 0 or moderately negative "
            "(as long as every value stays above -1) — useful for things like a bounded "
            "percentage change that log can't handle."
        ),
        formula_kind="latex",
        formula=r"x' = \ln(1 + x)",
        build_demo=_demo_pct_change_bounded,
        edge_cases=[
            "Every value must be greater than -1 — a column with values ≤ -1 (e.g. an "
            "unbounded cash-flow amount, not a percentage) will fail to fit. Use yeo_johnson "
            "instead for those.",
            "A value outside the range seen during fit becomes a blank/missing value at "
            "scoring time rather than an error.",
        ],
    ),
    GlossaryEntry(
        name="sqrt",
        category="Distribution shape",
        what_it_does=(
            "A gentler compression than log — pulls in large values without flattening them as "
            "aggressively. A common choice for count-like data (number of transactions, days "
            "since an event) where log would over-compress."
        ),
        formula_kind="latex",
        formula=r"x' = \sqrt{x}",
        build_demo=_demo_balance_positive,
        edge_cases=[
            "Every value must be non-negative — a column with negative values will fail to fit. "
            "Use yeo_johnson instead.",
            "A negative value at scoring time (out of the training range) becomes a blank/"
            "missing value rather than an error.",
        ],
    ),
    GlossaryEntry(
        name="cbrt",
        category="Distribution shape",
        what_it_does=(
            "Like sqrt, but works on negative values too (a cube root of a negative number is "
            "still a real number) — useful for a signed quantity like net cash flow that can be "
            "positive or negative but still benefits from compression."
        ),
        formula_kind="latex",
        formula=r"x' = \sqrt[3]{x}",
        build_demo=_demo_cashflow_with_zero_negative,
        edge_cases=[
            "No domain restriction — unlike sqrt or log, works on any real number including "
            "negative values and zero.",
            "Compresses large values *more* aggressively than sqrt, not less — cbrt(1000) = 10 "
            "vs. sqrt(1000) ≈ 31.6, since a smaller root exponent pulls large numbers in harder.",
        ],
    ),
    GlossaryEntry(
        name="reciprocal",
        category="Distribution shape",
        what_it_does=(
            "Flips a column onto an inverse scale (1/x) — turns a 'rate' into a 'time', or "
            "vice versa (e.g. transactions-per-day becomes days-per-transaction). Large values "
            "shrink toward zero; small values close to zero blow up."
        ),
        formula_kind="latex",
        formula=r"x' = \frac{1}{x}",
        build_demo=_demo_balance_positive,
        edge_cases=[
            "Every value must be nonzero — a column containing exactly 0 will fail to fit, "
            "since 1/0 is undefined.",
            "A value very close to zero produces a very large output — this transform is "
            "unstable near zero even when technically defined.",
        ],
    ),
    GlossaryEntry(
        name="yeo_johnson",
        category="Distribution shape",
        what_it_does=(
            "A more flexible version of log/log1p that automatically learns the best power "
            "transform for each column and works natively on zero and negative values — the "
            "right default for skewed banking numerics like balance or income when you're not "
            "sure the column is always positive."
        ),
        formula_kind="text",
        formula=(
            'Fits scikit-learn\'s PowerTransformer(method="yeo-johnson") per column: a '
            "maximum-likelihood-fitted power transform with no positivity requirement."
        ),
        build_demo=_demo_cashflow_with_zero_negative,
        edge_cases=[
            "Works on any real number — zero, negative, or positive — unlike log/log1p.",
            "The exact transform (its fitted parameter) is learned from training data and "
            "reused as-is at scoring time.",
        ],
    ),
    GlossaryEntry(
        name="quantile_uniform",
        category="Distribution shape",
        what_it_does=(
            "Replaces each value with its rank in the training distribution, spread evenly "
            "between 0 and 1 — makes every column look uniformly distributed, regardless of "
            "its original shape."
        ),
        formula_kind="text",
        formula="Rank-based mapping to a uniform [0, 1] distribution via training quantiles.",
        build_demo=_demo_income_with_outlier,
        edge_cases=[
            "Extreme outliers get squeezed to the same [0, 1] range as everything else — this "
            "is one of the most outlier-resistant transforms available.",
            "A value below the smallest (or above the largest) training value gets clipped to "
            "0 (or 1) at scoring time.",
        ],
    ),
    GlossaryEntry(
        name="quantile_normal",
        category="Distribution shape",
        what_it_does=(
            "Same idea as quantile_uniform, but maps ranks onto a standard normal (bell-curve) "
            "distribution instead of a uniform one — useful when a downstream model or chart "
            "assumes roughly normal inputs."
        ),
        formula_kind="text",
        formula="Rank-based mapping to a standard normal distribution via training quantiles.",
        build_demo=_demo_income_with_outlier,
        edge_cases=[
            "Same outlier-resistance as quantile_uniform, just on a different output scale.",
            "A value below/above the training range gets mapped to the same extreme as the "
            "training minimum/maximum.",
        ],
    ),
    GlossaryEntry(
        name="bucket_quantile",
        category="Binning",
        what_it_does=(
            "Splits a numeric column into equal-frequency buckets (roughly the same number of "
            "rows in each bucket) — turns a continuous column like age into a small number of "
            "groups."
        ),
        formula_kind="text",
        formula='Equal-frequency bins via scikit-learn\'s KBinsDiscretizer(strategy="quantile").',
        build_demo=_demo_age,
        demo_params={"n_bins": 3},
        edge_cases=[
            "A column with only one distinct value can't be usefully binned and is skipped.",
            "A value outside the bin edges learned at fit time (or a missing value) is coded "
            'as "unknown" at scoring time.',
        ],
    ),
    GlossaryEntry(
        name="bucket_uniform",
        category="Binning",
        what_it_does=(
            "Splits a numeric column into equal-width buckets (each bucket covers the same "
            "range of values, regardless of how many rows fall in it)."
        ),
        formula_kind="text",
        formula='Equal-width bins via scikit-learn\'s KBinsDiscretizer(strategy="uniform").',
        build_demo=_demo_age,
        demo_params={"n_bins": 3},
        edge_cases=[
            "If most rows cluster in one range, bins can end up very unevenly populated (unlike "
            "bucket_quantile, which always balances row counts).",
            'A value outside the bin edges learned at fit time is coded as "unknown".',
        ],
    ),
    GlossaryEntry(
        name="bucket_tree",
        category="Binning",
        what_it_does=(
            "Splits a numeric column into buckets chosen by a small decision tree trained "
            "against your target — the bucket boundaries are wherever the tree found the "
            "target most separable, not just equal-count or equal-width splits."
        ),
        formula_kind="text",
        formula=(
            "Split points = a decision tree's learned thresholds, fit on this column vs. the "
            "target."
        ),
        build_demo=_demo_age_with_target,
        edge_cases=[
            "Needs a target (y) to fit — falls back to bucket_quantile's behavior when no "
            "target is available.",
            "Bucket boundaries can look arbitrary compared to bucket_quantile/bucket_uniform, "
            "since they're chosen to separate the target, not to balance row counts.",
        ],
    ),
    GlossaryEntry(
        name="percentile_rank",
        category="Ranking & scaling",
        what_it_does=(
            "Replaces each value with the fraction of training rows it's higher than — a "
            "customer with the highest income in training gets close to 1.0, the lowest gets "
            "close to 0.0. Neutralises the influence of extreme outliers without discarding "
            "their relative order."
        ),
        formula_kind="latex",
        formula=r"x' = \frac{\#\{\,x_{train} < x\,\}}{n}",
        build_demo=_demo_income_with_outlier,
        edge_cases=[
            "A value below every training value becomes 0.0; above every training value "
            "becomes 1.0 — no errors on out-of-range values, unlike log/log1p.",
            "Missing values stay missing.",
        ],
    ),
    GlossaryEntry(
        name="dense_rank",
        category="Ranking & scaling",
        what_it_does=(
            "Like percentile_rank, but counts distinct values rather than rows — two customers "
            "with the exact same income get the same dense rank, and the next distinct income "
            "up gets the very next rank with no gap. Useful when you care about 'how many "
            "different values came before this one', not raw row counts."
        ),
        formula_kind="text",
        formula=(
            "Index of the value among the sorted *unique* training values (0-based, ties "
            "share a rank)."
        ),
        build_demo=_demo_income_with_outlier,
        edge_cases=[
            "Unlike percentile_rank, the output isn't normalised to [0, 1] — it's a raw count "
            "of distinct values, so its scale depends on how many unique values training had.",
            "Missing values stay missing.",
        ],
    ),
    GlossaryEntry(
        name="global_rank",
        category="Ranking & scaling",
        what_it_does=(
            "The simplest ordinal ranking: counts how many training rows are at or below each "
            "value, ties included — a customer tied for the highest income still counts every "
            "row at that income level, not just their own."
        ),
        formula_kind="latex",
        formula=r"x' = \#\{\,x_{train} \leq x\,\}",
        build_demo=_demo_income_with_outlier,
        edge_cases=[
            "Not normalised — the output ranges from 0 up to the training row count, unlike "
            "percentile_rank's [0, 1] scale.",
            "Ties are counted together: if three rows share the highest value, all three get "
            "the same rank equal to the full row count.",
        ],
    ),
    GlossaryEntry(
        name="scale",
        category="Ranking & scaling",
        what_it_does=(
            "Rescales a numeric column so its values sit on a comparable range to other "
            "features — by default using the median and interquartile range (robust to "
            "outliers), rather than mean/standard deviation."
        ),
        formula_kind="latex",
        formula=r"x' = \frac{x - \mathrm{median}(x_{train})}{\mathrm{IQR}(x_{train})}",
        build_demo=_demo_income_with_outlier,
        edge_cases=[
            "Missing values are filled with 0 before scaling, as a side effect — this is not "
            "configurable.",
            'The default ("robust") strategy is deliberately outlier-resistant; switching to '
            '"standard" (mean/std) would let the one extreme value in this demo dominate the '
            "result.",
        ],
    ),
    GlossaryEntry(
        name="ordinal_encode",
        category="Categorical encoding",
        what_it_does=(
            "Assigns each category a whole-number code (0, 1, 2, ...) — simple and compact, "
            "but implies an order between categories that may not actually exist, so it's best "
            "for genuinely ordered categories (e.g. risk tiers) rather than unordered ones."
        ),
        formula_kind="code",
        formula="code = alphabetical_rank(category)   # or a user-supplied order",
        build_demo=_demo_segment,
        edge_cases=[
            "A category never seen during fitting (or a missing value) is coded as -1 at "
            "scoring time.",
        ],
    ),
    GlossaryEntry(
        name="onehot_encode",
        category="Categorical encoding",
        what_it_does=(
            "Creates one new 0/1 column per category — the standard, order-free way to encode "
            "a categorical column, at the cost of adding one column per distinct value."
        ),
        formula_kind="code",
        formula="{column}_{category} = 1 if row's value == category else 0",
        build_demo=_demo_segment,
        edge_cases=[
            "The number of output columns is fixed at fit time — a category never seen during "
            "fitting produces an all-zero row rather than a new column.",
            "High-cardinality columns can produce a lot of columns — consider rare_group first, "
            "or use target_encode/frequency_encode instead.",
        ],
    ),
    GlossaryEntry(
        name="rare_group",
        category="Categorical encoding",
        what_it_does=(
            'Collapses categories that appear too rarely into a single "Other" bucket — '
            "reduces noise from one-off categories before they reach an encoder or model."
        ),
        formula_kind="text",
        formula=(
            "Keep categories with row-share ≥ min_frequency; replace everything else with "
            '"Other".'
        ),
        build_demo=_demo_segment_with_rare,
        demo_params={"min_frequency": 0.2},
        edge_cases=[
            "Doesn't change the column type or count — it's a cleanup step meant to run before "
            "an encoder (ordinal/one-hot/target/WoE), not a replacement for one.",
            'A category never seen during fitting is also grouped into "Other" at scoring '
            "time, the same as a rare one.",
            "This demo lowers the rarity threshold to 20% purely to make it visible on 7 rows — "
            "the real default is 1%.",
        ],
    ),
    GlossaryEntry(
        name="target_encode",
        category="Categorical encoding",
        what_it_does=(
            "Replaces each category with a smoothed estimate of how likely the target is for "
            "that category — e.g. a branch's historical default rate, pulled toward the "
            "overall average when the branch has few observations. Only applied to columns "
            "with many distinct values (high cardinality)."
        ),
        formula_kind="latex",
        formula=r"x'_{cat} = \frac{\sum y_{in\ cat} + m \cdot \bar y}{n_{in\ cat} + m}",
        build_demo=_demo_high_cardinality_with_target,
        demo_params={"cardinality_threshold": 3},
        edge_cases=[
            "Needs a target (y) to fit — without one, it silently falls back to "
            "frequency_encode's behavior instead.",
            "A category never seen during fitting is filled with the overall training target "
            "average at scoring time.",
            "Only processes columns with more distinct values than the cardinality threshold "
            "(default 50) — this demo lowers it to 3 to trigger on a small example.",
        ],
    ),
    GlossaryEntry(
        name="frequency_encode",
        category="Categorical encoding",
        what_it_does=(
            "Replaces each category with how often it appears in the training data (as a "
            "fraction of all rows) — doesn't need a target, unlike target_encode."
        ),
        formula_kind="latex",
        formula=r"x'_{cat} = \frac{\mathrm{count}(cat)}{n}",
        build_demo=_demo_high_cardinality_with_target,
        demo_params={"cardinality_threshold": 3},
        edge_cases=[
            "A category never seen during fitting is filled with 1/n_unique at scoring time.",
            "Only processes columns with more distinct values than the cardinality threshold "
            "(default 50) — this demo lowers it to 3 to trigger on a small example.",
        ],
    ),
    GlossaryEntry(
        name="woe_encode",
        category="Categorical encoding",
        what_it_does=(
            "Replaces each category (or numeric bucket) with how much more or less likely the "
            "outcome is for rows in that group, compared to the overall average — a standard "
            "credit-scoring technique that also produces an Information Value ranking the "
            "column's predictive power."
        ),
        formula_kind="latex",
        formula=(
            r"WoE = \ln\!\left(\frac{\%\ events\ in\ bin}{\%\ non\text{-}events\ in\ bin}\right)"
        ),
        build_demo=_demo_segment_with_target,
        edge_cases=[
            "Needs a strict 0/1 target — will not run on a continuous or multi-class target.",
            "WoE values are capped at ±4 to avoid infinities when a bin has only good or only "
            "bad outcomes.",
            "A category never seen during fitting gets a neutral WoE of 0 at scoring time.",
        ],
    ),
    GlossaryEntry(
        name="binary_encode",
        category="Categorical encoding",
        what_it_does=(
            "Encodes categories as the binary digits of an ordinal code — a column with 8 "
            "categories needs only 3 output columns instead of 8 (as one-hot would need), "
            "trading a little interpretability for far fewer columns on high-cardinality data."
        ),
        formula_kind="code",
        formula=(
            "index = alphabetical_rank(category)\n"
            "n_bits = ceil(log2(n_categories))\n"
            "{column}_bin{i} = bit i of index"
        ),
        build_demo=_demo_segment,
        edge_cases=[
            "A category never seen during fitting produces all-zero bit columns at scoring "
            "time.",
            "Unlike one-hot, the encoded values have no simple 1-to-1 meaning per column — a "
            "bit column doesn't correspond to one specific category.",
        ],
    ),
    GlossaryEntry(
        name="hash_encode",
        category="Categorical encoding",
        what_it_does=(
            'Encodes categories via the "hashing trick": each value is hashed into one of a '
            "fixed number of buckets. Needs no memory of which categories were seen at fit "
            "time (unlike every other encoder here), at the cost of a small, accepted chance "
            "two different categories land in the same bucket."
        ),
        formula_kind="code",
        formula=(
            "digest = md5(value)\n"
            "bucket = int(digest[:8], 16) % n_components\n"
            "sign = +1 or -1, derived from the digest\n"
            "{column}_hash{bucket} = sign"
        ),
        build_demo=_demo_segment,
        edge_cases=[
            "Stateless — a brand-new category at scoring time hashes to a bucket exactly like "
            'any other value, no "unseen category" case to handle.',
            "Two unrelated categories can collide into the same bucket (a known, accepted "
            "trade-off for the fixed, small column count).",
        ],
    ),
    GlossaryEntry(
        name="polynomial",
        category="Feature expansion",
        what_it_does=(
            "Generates squared terms and pairwise-interaction terms from numeric columns — "
            "e.g. from age and income, adds age², age×income, and income², letting a linear "
            "model capture some non-linear/interaction effects."
        ),
        formula_kind="code",
        formula="For degree=2 inputs x1, x2 → x1, x2, x1², x1·x2, x2²  (no intercept column)",
        build_demo=_demo_age_income,
        edge_cases=[
            "Column-expanding — the number of output columns grows quickly with the number of "
            "input columns and the chosen degree.",
            "Needs at least one numeric column to run.",
        ],
    ),
    GlossaryEntry(
        name="spline",
        category="Feature expansion",
        what_it_does=(
            "Expands a numeric column into a small set of smooth curve segments (a B-spline "
            "basis), letting a linear model fit smooth non-linear effects without manually "
            "choosing bucket boundaries."
        ),
        formula_kind="text",
        formula=(
            "scikit-learn's SplineTransformer(n_knots, degree) — a smooth piecewise-polynomial "
            "basis."
        ),
        build_demo=_demo_age,
        edge_cases=[
            "Column-expanding, similar to one-hot or polynomial — each input column becomes "
            "several output columns.",
            "Needs at least one numeric column with enough distinct values to place knots "
            "sensibly.",
        ],
    ),
    GlossaryEntry(
        name="noise",
        category="Data augmentation",
        what_it_does=(
            "Adds small random perturbations to training data only — never to validation, "
            "test, or scoring data — as a way to make a model less likely to memorise exact "
            "training values. Numeric columns get small additive noise scaled to the column's "
            "own spread; categorical columns occasionally get swapped to another category "
            "that was actually observed in training."
        ),
        formula_kind="latex",
        formula=(
            r"x'_{numeric} = x + \mathcal{N}\bigl(0,\ scale \times \mathrm{std}(x_{train})\bigr)"
        ),
        build_demo=_demo_balance_positive,
        demo_params={"affected_row_frac": 1.0, "random_state": 42},
        edge_cases=[
            "Only affects the training set — every other split passes through completely "
            "unchanged, by design.",
            "This demo sets affected_row_frac to 100% so every row visibly changes; the real "
            "default only perturbs about 5% of rows.",
            "A column with zero variance (every row the same value) has nothing to scale noise "
            "by and is skipped.",
        ],
    ),
]

if {e.name for e in _ENTRIES} != set(TRANSFORMER_REGISTRY):
    raise AssertionError(
        "Every TRANSFORMER_REGISTRY key must have exactly one GlossaryEntry, and vice versa "
        "— add/remove an entry above to match."
    )

_ENTRIES_BY_NAME: dict[str, GlossaryEntry] = {e.name: e for e in _ENTRIES}
_NAMES_BY_CATEGORY: dict[str, list[str]] = {
    category: sorted(e.name for e in _ENTRIES if e.category == category) for category in _CATEGORIES
}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_formula(entry: GlossaryEntry) -> None:
    if entry.formula_kind == "latex":
        st.latex(entry.formula)
    elif entry.formula_kind == "code":
        st.code(entry.formula, language="text")
    else:
        st.markdown(entry.formula)


def _render_example(entry: GlossaryEntry) -> None:
    X_demo, y_demo = entry.build_demo()
    try:
        transformer = TRANSFORMER_REGISTRY[entry.name](**entry.demo_params)
        X_after = transformer.fit_transform(X_demo, y_demo)
    except Exception as exc:
        logger.exception("Glossary demo failed for %s", entry.name)
        st.error(f"Couldn't build the live example for this transform: {exc}")
        return

    col_before, col_after = st.columns(2)
    with col_before:
        st.caption("Before")
        st.dataframe(X_demo, hide_index=True, width="stretch")
    with col_after:
        st.caption("After")
        st.dataframe(X_after, hide_index=True, width="stretch")
    st.caption("Computed live using the same code the pipeline runs — not a static example.")


def render_glossary() -> None:
    """Render the Glossary page: category picker → transform picker → detail.

    Args:
        None

    Returns:
        None
    """
    st.subheader("Glossary: feature-processing transformations")
    st.caption(
        "What each Step 5 recipe option does, its formula, and a live worked example — pick a "
        "category, then a transformation, to see the details."
    )

    category = st.segmented_control(
        "Category",
        _CATEGORIES,
        default=_CATEGORIES[0],
        key="glossary.category",
        label_visibility="collapsed",
    )
    if category is None:
        st.info("Pick a category above to get started.")
        return

    names = _NAMES_BY_CATEGORY[category]
    name = st.selectbox("Transformation", names, key="glossary.transform")
    entry = _ENTRIES_BY_NAME[name]

    st.markdown(f"### `{entry.name}`")
    st.write(entry.what_it_does)

    st.markdown("**Formula**")
    _render_formula(entry)

    st.markdown("**Example**")
    _render_example(entry)

    st.markdown("**Things to know / edge cases**")
    for case in entry.edge_cases:
        st.write(f"- {case}")
