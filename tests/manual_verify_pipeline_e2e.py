"""Manual verification script — full PipelineRunner smoke test on synthetic data.

Not a pytest-collected test (see ``manual_verify_interactive_flow.py`` for the same
convention). Written to answer a concrete question: "is dscompanion actually ready to run
end-to-end on a fresh environment?" — a quick core-library smoke test: import
dscompanion, run ``DataSplitter`` on a small synthetic dataset, run
``PipelineRunner`` end-to-end. Doubles as a notebook-paste-able cell for any
restricted or offline environment — no dependency on this repo's test
fixtures or pytest itself, just ``dscompanion`` + a temp directory.

Uses dscompanion's own ``SyntheticDataGenerator`` (dscompanion.utils.synthetic) rather than
sklearn's ``make_classification`` — it produces the *messy* feature characteristics
(missing values, near-zero variance, constant columns, high-cardinality categoricals)
this pipeline is actually built to handle, not just clean separable numeric features.
``make_classification`` would only exercise the "everything is already clean" path.

Two runs, back to back:
1. Baseline — today's default global per-dtype feature processing
   (SmartImputer + FeatureProcessingPipeline), no per-column overrides.
2. Recipe-based — same dataset, but a couple of columns get an explicit
   ``ColumnRecipe`` via the new ``feature_recipes`` field —
   the newest, least battle-tested code path, so it's worth exercising here
   specifically rather than assuming the unit tests alone are enough.

Both runs confirm the Excel model card (the primary artefact) and the HTML report (opt-in via
reporting.html_report) are both produced correctly from the same ModelCard object.

Prints Python + tracked third-party package versions up front (see
``TRACKED_PACKAGES``), so a failure in a new environment can be diffed
directly against a known-good pip-freeze snapshot without a separate
round-trip.

Run with (from dscompanion/ as cwd)::

    conda activate dscompanion312 && python tests/manual_verify_pipeline_e2e.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

from dscompanion.features import ColumnRecipe
from dscompanion.pipeline import PipelineConfig, PipelineRunner
from dscompanion.pipeline.config import DataConfig, ExplainConfig, ModelConfig, ReportingConfig
from dscompanion.utils.synthetic import SyntheticDataGenerator

N_ROWS = 5_000
N_NUMERIC = 12
N_CATEGORICAL = 6
RANDOM_STATE = 42

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
    "shap",
    "pydantic",
    "pydantic-settings",
    "statsmodels",
    "optuna",
    "joblib",
    "matplotlib",
    "seaborn",
    "plotly",
    "python-docx",
    "xlsxwriter",
]


def print_environment_info() -> None:
    """Print Python + tracked-package versions up front, before anything can fail.

    Args:
        None

    Returns:
        None
    """
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")  # noqa: T201
    for name in TRACKED_PACKAGES:
        try:
            print(f"  {name}=={version(name)}")  # noqa: T201
        except PackageNotFoundError:
            print(f"  {name}==NOT INSTALLED")  # noqa: T201


def build_dataset() -> "pd.DataFrame":
    """Generate messy synthetic features, then attach a genuine binary target.

    The target is derived from two "normal" numeric columns and one "normal"
    categorical column's most frequent level — real signal, not pure noise —
    plus Gaussian noise, so downstream metrics land meaningfully above chance
    rather than at a perfect (suspicious) or random (useless) score. For a
    more domain-realistic signal (e.g. mirroring an actual response
    pattern), this is the one block worth swapping — everything below it
    works off whatever "target" column comes out.

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

    print(  # noqa: T201
        f"Target driven by: {driver_num_a}, {driver_num_b}, {driver_cat}=='{top_category}'"
    )
    print(f"Target event rate: {df['target'].mean():.3f}")  # noqa: T201
    return df


def run_baseline(parquet_path: Path, output_dir: Path):
    """Today's default feature-processing path — no feature_recipes."""
    cfg = PipelineConfig(
        name="e2e_smoke_baseline",
        data=DataConfig(path=str(parquet_path), target="target"),
        model=ModelConfig(task="classification", algorithm="xgboost"),
        explain=ExplainConfig(shap_enabled=False),
        reporting=ReportingConfig(output_dir=str(output_dir), html_report=True),
    )
    return PipelineRunner(cfg).run()


def build_recipes_for_every_column(df: "pd.DataFrame") -> dict[str, ColumnRecipe]:
    """Recommend a ColumnRecipe for every feature column — mirrors Step 5's
    "Accept all recommended" bulk action (interactive_step_feature_processing.py),
    not a hand-picked subset. A categorical column left without *some* encoding
    step reaches XGBoost as a raw string and crashes — every column needs a
    recipe, not just the ones a human happens to pick for a quick manual test.

    Args:
        df: The full synthetic DataFrame, including the "target" column
            (excluded from the returned recipes).

    Returns:
        dict[str, ColumnRecipe]: One recommended recipe per feature column.
    """
    from dscompanion.eda.univariate import UnivariateAnalyser
    from dscompanion.features import recommend_column_recipe
    from dscompanion.split import DataSplitter

    split = DataSplitter(strategy="stratified", target_col="target").fit_split(df)
    analyser = UnivariateAnalyser().fit(split)
    numeric_summary = analyser.numeric_summary()
    categorical_summary = analyser.categorical_summary()

    recipes: dict[str, ColumnRecipe] = {}
    for col in df.columns:
        if col == "target":
            continue
        is_numeric = col in set(numeric_summary["feature"])
        summary = numeric_summary if is_numeric else categorical_summary
        stats = summary.loc[summary["feature"] == col].iloc[0]
        recipes[col] = recommend_column_recipe(
            col, stats, "numeric" if is_numeric else "categorical"
        )
    return recipes


def run_with_recipes(parquet_path: Path, df: "pd.DataFrame", output_dir: Path):
    """Same dataset, exercising the new ColumnRecipe/FeatureTransformChain path —
    every column gets its recommended recipe, same as Step 5's bulk-accept flow.
    """
    recipes = build_recipes_for_every_column(df)
    cfg = PipelineConfig(
        name="e2e_smoke_recipes",
        data=DataConfig(path=str(parquet_path), target="target"),
        feature_recipes=recipes,
        model=ModelConfig(task="classification", algorithm="xgboost"),
        explain=ExplainConfig(shap_enabled=False),
        reporting=ReportingConfig(output_dir=str(output_dir), html_report=True),
    )
    return PipelineRunner(cfg).run()


def summarize(label: str, result, output_dir: Path) -> None:
    gini_row = result.metrics.query("split == 'test' and metric == 'gini'")
    auc_row = result.metrics.query("split == 'test' and metric == 'roc_auc'")
    print(f"\n=== {label} ===")  # noqa: T201
    print(f"  elapsed: {result.elapsed_seconds:.1f}s")  # noqa: T201
    print(f"  features after selection: {result.split.train_X.shape[1]}")  # noqa: T201
    print(f"  test gini: {float(gini_row['value'].iloc[0]):.4f}")  # noqa: T201
    print(f"  test roc_auc: {float(auc_row['value'].iloc[0]):.4f}")  # noqa: T201
    xlsx_path = output_dir / f"{result.config.name}_model_card.xlsx"
    result.model_card.to_excel(xlsx_path)
    print(f"  excel model card: {xlsx_path}")  # noqa: T201
    print(f"  html report: {result.report_path}")  # noqa: T201
    assert result.model is not None
    assert result.report_path is not None and Path(result.report_path).exists()
    assert xlsx_path.exists() and xlsx_path.stat().st_size > 0
    assert 0.0 <= float(gini_row["value"].iloc[0]) <= 1.0
    assert float(auc_row["value"].iloc[0]) > 0.5, "model should beat chance on real signal"


def main() -> None:
    t_start = time.perf_counter()
    print_environment_info()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        parquet_path = tmp_path / "synthetic.parquet"
        output_dir = tmp_path / "reports"
        output_dir.mkdir()

        df = build_dataset()
        df.to_parquet(parquet_path, index=False)
        print(f"Synthetic dataset written: {df.shape} -> {parquet_path}")  # noqa: T201

        baseline_result = run_baseline(parquet_path, output_dir)
        summarize("Baseline (default feature processing)", baseline_result, output_dir)

        recipe_result = run_with_recipes(parquet_path, df, output_dir)
        summarize("ColumnRecipe / FeatureTransformChain path", recipe_result, output_dir)

    print(f"\nALL CHECKS PASSED. Total time: {time.perf_counter() - t_start:.1f}s")  # noqa: T201


if __name__ == "__main__":
    main()
