"""Manual verification script — Leaderboard + Tuner smoke test on synthetic data.

Not a pytest-collected test (see ``manual_verify_pipeline_e2e.py`` for the same
convention). Written to confirm dscompanion's existing multi-algorithm comparison
(``Leaderboard``) and hyperparameter search (``Tuner``) work correctly together
against real data — both classes are already unit-tested locally and have
zero MLflow dependency, but neither has been exercised together end-to-end
outside a unit-test context before. Doubles as a notebook-paste-able cell
for any restricted or offline environment — no dependency on this repo's
test fixtures or pytest itself, just ``dscompanion`` + a temp directory.

Two stages, back to back:
1. Leaderboard — compare every supported classification algorithm on one
   synthetic dataset, rank by ``roc_auc``, identify the champion.
2. Tuner — take the champion algorithm's fitted model and run a bounded
   Optuna search (``n_trials`` kept small — this is a smoke test, not a
   real tuning budget) to confirm tuning actually improves (or at least
   doesn't break) on top of the untuned champion.

Prints Python + tracked third-party package versions up front (see
``TRACKED_PACKAGES``), so a failure in a new environment can be diffed
directly against a known-good pip-freeze snapshot without a separate
round-trip.

Run with (from dscompanion/ as cwd)::

    conda activate dscompanion312 && python tests/manual_verify_leaderboard_tuning.py
"""

from __future__ import annotations

import sys
import time
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd

from dscompanion.utils.synthetic import SyntheticDataGenerator

N_ROWS = 5_000
N_NUMERIC = 12
N_CATEGORICAL = 6
RANDOM_STATE = 42
N_TUNING_TRIALS = 20

# Distribution-package name (importlib.metadata key), not the import name —
# captured up front so a failure in a new environment can be diffed
# directly against a known-good pip-freeze snapshot without a separate
# `pip freeze` round-trip.
TRACKED_PACKAGES = [
    "pandas",
    "numpy",
    "scikit-learn",
    "scipy",
    "xgboost",
    "optuna",
    "pydantic",
    "pydantic-settings",
    "joblib",
]


def print_environment_info() -> None:
    """Print Python + tracked-package versions up front, before anything can fail.

    Args:
        None

    Returns:
        None
    """
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    for name in TRACKED_PACKAGES:
        try:
            print(f"  {name}=={version(name)}")
        except PackageNotFoundError:
            print(f"  {name}==NOT INSTALLED")


def build_dataset() -> pd.DataFrame:
    """Generate messy synthetic features, then attach a genuine binary target.

    The target is derived from two "normal" numeric columns and one "normal"
    categorical column's most frequent level — real signal, not pure noise —
    plus Gaussian noise, so downstream metrics land meaningfully above chance
    rather than at a perfect (suspicious) or random (useless) score. Mirrors
    ``manual_verify_pipeline_e2e.py``'s ``build_dataset()`` (same synthetic
    data recipe) — kept as a separate copy rather than a shared import,
    matching this file's own "no dependency on other repo files" design goal
    (must stay pasteable into a notebook cell with only ``dscompanion`` importable).

    Returns:
        pd.DataFrame: n_rows x (n_numeric + n_categorical + 1) — feature
        columns from SyntheticDataGenerator plus a binary "target" column.
    """
    gen = SyntheticDataGenerator(
        n_rows=N_ROWS,
        n_numeric=N_NUMERIC,
        n_categorical=N_CATEGORICAL,
        random_state=RANDOM_STATE,
    )
    df = gen.generate()

    rng = np.random.RandomState(RANDOM_STATE)
    numeric_normal_cols = sorted(c for c in df.columns if c.startswith("num_normal_"))
    categorical_normal_cols = sorted(c for c in df.columns if c.startswith("cat_normal_"))
    driver_num_a, driver_num_b = numeric_normal_cols[:2]
    driver_cat = categorical_normal_cols[0]

    top_category = df[driver_cat].value_counts(dropna=True).idxmax()
    signal = (
        0.02 * df[driver_num_a].fillna(df[driver_num_a].mean())
        - 0.015 * df[driver_num_b].fillna(df[driver_num_b].mean())
        + 1.2 * (df[driver_cat] == top_category).astype(float)
    )
    logit = signal - signal.mean() + rng.normal(0, 1.0, N_ROWS)
    prob = 1 / (1 + np.exp(-logit))
    df["target"] = (rng.uniform(0, 1, N_ROWS) < prob).astype(int)

    print(f"Target driven by: {driver_num_a}, {driver_num_b}, {driver_cat}=='{top_category}'")
    print(f"Target event rate: {df['target'].mean():.3f}")
    return df


def _process_split(split):
    """Impute + encode every partition of a raw DataSplit — Leaderboard needs numeric input.

    ``Leaderboard`` (and ``Tuner``, downstream) fit directly on
    ``split.X_train``/``split.y_train`` with no feature processing of their
    own — that is the caller's responsibility, same as this repo's own
    ``FeatureProcessingPipeline`` quickstart example. ``FeatureProcessingPipeline``
    is fit once on the training partition only (never on val/test/oot — fitting
    on held-out data would leak label-derived statistics into evaluation), then
    applied to every partition via ``transform()``.

    Args:
        split: Raw ``DataSplit`` from ``DataSplitter.fit_split()``, with
            possibly-categorical/possibly-missing feature columns.

    Returns:
        DataSplit: A new ``DataSplit`` with every ``*_X`` partition replaced
        by its processed (numeric-only, imputed) version; ``*_y`` and
        ``metadata`` are carried over unchanged.
    """
    from dscompanion.features import FeatureProcessingPipeline
    from dscompanion.split import DataSplit

    fp = FeatureProcessingPipeline()
    fp.fit(split.train_X, split.train_y)

    def _transform_or_empty(X: pd.DataFrame) -> pd.DataFrame:
        # Some split strategies (e.g. "stratified") leave oot_X empty (zero
        # rows) — transforming zero rows through the scaler raises inside
        # sklearn's check_array. Nothing downstream evaluates on oot here,
        # so just pass the empty frame through untouched.
        return fp.transform(X) if len(X) > 0 else X

    return DataSplit(
        train_X=fp.transform(split.train_X),
        train_y=split.train_y,
        val_X=_transform_or_empty(split.val_X),
        val_y=split.val_y,
        test_X=_transform_or_empty(split.test_X),
        test_y=split.test_y,
        oot_X=_transform_or_empty(split.oot_X),
        oot_y=split.oot_y,
        metadata=split.metadata,
    )


def run_leaderboard(df: pd.DataFrame):
    """Split the dataset, process features, run every supported algorithm, and rank them.

    Args:
        df: Full synthetic DataFrame including the "target" column, as
            returned by ``build_dataset()``.

    Returns:
        tuple: ``(leaderboard, split)`` — the fitted ``Leaderboard`` instance
        (``.leaderboard_`` and ``.fitted_models_`` populated) and the
        **processed** ``DataSplit`` it was run on, so the tuning step can
        reuse the exact same (already-numeric) train/val partition rather
        than re-splitting or re-processing.
    """
    from dscompanion.leaderboard import Leaderboard
    from dscompanion.split import DataSplitter

    raw_split = DataSplitter(strategy="stratified", target_col="target").fit_split(df)
    split = _process_split(raw_split)

    leaderboard = Leaderboard(task="classification", sort_metric="roc_auc", ascending=False)
    leaderboard.run(split)

    print("\n=== Leaderboard ===")
    print(leaderboard.leaderboard_.to_string(index=False))

    n_ok = (leaderboard.leaderboard_["status"] == "ok").sum()
    n_total = len(leaderboard.leaderboard_)
    print(f"\n{n_ok}/{n_total} algorithms trained successfully")

    assert n_ok > 0, "every algorithm failed — leaderboard produced no usable candidates"
    assert leaderboard.fitted_models_, "fitted_models_ is empty despite successful rows"

    champion_name = leaderboard.leaderboard_.iloc[0]["algorithm"]
    champion_auc = leaderboard.leaderboard_.iloc[0]["roc_auc"]
    print(f"Champion: {champion_name} (roc_auc={champion_auc:.4f})")
    assert 0.5 < champion_auc <= 1.0, "champion should beat chance on real signal"

    return leaderboard, split


def run_tuning(leaderboard, split) -> None:
    """Tune the leaderboard's champion algorithm and compare against its untuned score.

    Args:
        leaderboard: The fitted ``Leaderboard`` instance from
            ``run_leaderboard()`` — its ``.leaderboard_`` (ranked results)
            and ``.fitted_models_`` (algorithm name -> fitted model) are
            both read here.
        split: The (already-processed) ``DataSplit`` ``run_leaderboard()``
            was run on — reused here as ``Tuner``'s ``cv`` argument so
            tuning scores against the exact same held-out partition the
            leaderboard ranked on.

    Returns:
        None. Prints the tuning outcome and asserts basic sanity (tuning
        completed, produced a real best_score_, and the tuned model still
        beats chance).
    """
    from dscompanion.tuning import Tuner

    champion_name = leaderboard.leaderboard_.iloc[0]["algorithm"]
    champion_model = leaderboard.fitted_models_[champion_name]
    untuned_auc = leaderboard.leaderboard_.iloc[0]["roc_auc"]

    print(f"\n=== Tuning champion: {champion_name} ({N_TUNING_TRIALS} trials) ===")
    t0 = time.perf_counter()
    tuner = Tuner(
        model=champion_model,
        backend="optuna",
        n_trials=N_TUNING_TRIALS,
        cv=split,
        metric="roc_auc",
        direction="maximize",
    )
    tuned_model = tuner.run()
    elapsed = time.perf_counter() - t0

    print(f"Tuning elapsed: {elapsed:.1f}s")
    print(f"Best params: {tuner.best_params_}")
    print(f"Best score (Optuna's internal val score): {tuner.best_score_:.4f}")

    tuned_metrics = tuned_model.evaluate(split)
    tuned_auc_row = tuned_metrics.query("split == 'test' and metric == 'roc_auc'")
    tuned_auc = float(tuned_auc_row["value"].iloc[0])
    print(f"Untuned {champion_name} test roc_auc: {untuned_auc:.4f}")
    print(f"Tuned   {champion_name} test roc_auc: {tuned_auc:.4f}")

    assert tuner.best_params_, "Tuner produced no best_params_"
    assert not np.isnan(tuner.best_score_), "Tuner's best_score_ is NaN"
    assert tuned_auc > 0.5, "tuned model should beat chance on real signal"


def main() -> None:
    t_start = time.perf_counter()
    print_environment_info()

    df = build_dataset()
    print(f"Synthetic dataset generated: {df.shape}")

    leaderboard, split = run_leaderboard(df)
    run_tuning(leaderboard, split)

    print(f"\nALL CHECKS PASSED. Total time: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
