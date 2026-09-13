# dscompanion Notebooks

Hands-on notebooks that teach `dscompanion` using real datasets from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/) — one notebook per
supported ML task, each driving the actual library end-to-end against live data. Every
dataset is fetched fresh at run time; nothing is bundled in this repo.

## Prerequisites

```bash
pip install dscompanion[notebooks]
```

This installs `dscompanion` itself plus `ucimlrepo` (the official UCI fetch client),
`jupyter`, and `ipykernel`. You'll also need internet access — every notebook pulls its
dataset live from UCI's servers on first run.

## Which notebook to start with

| Notebook | Dataset | Task | What it teaches beyond the others |
|---|---|---|---|
| [`01_getting_started.ipynb`](01_getting_started.ipynb) | Bank Marketing | Classification | The fastest path to a working `PipelineRunner.run()` — minimal config, minimal explanation, just to see it work |
| [`02_classification_bank_marketing.ipynb`](02_classification_bank_marketing.ipynb) | Bank Marketing | Classification | The deepest tour: EDA reports, splitting strategy, the leaderboard (multi-algorithm comparison), hyperparameter tuning, SHAP + permutation importance, and exporting a governance-ready model card |
| [`03_regression_wine_quality.ipynb`](03_regression_wine_quality.ipynb) | Wine Quality | Regression | Regression-specific evaluation metrics, and why some features (like the leaderboard) that classification gets aren't available for every task |
| [`04_clustering_wholesale_customers.ipynb`](04_clustering_wholesale_customers.ipynb) | Wholesale Customers | Clustering | Unsupervised evaluation (internal validity metrics, no ground-truth label), and a config quirk worth understanding: why a `target` column is still required even when there's nothing to predict |

If you only run one notebook, start with `01`. If you want the full picture of what
`dscompanion` can do, `02` is the one to read closely — `03` and `04` build on it rather
than repeating it.

## About the data cache

Each notebook fetches its dataset via `ucimlrepo` on first run and saves it locally as a
parquet file under `notebooks/.data_cache/` (created automatically). This directory is
gitignored — it's a local cache, not part of the repo, and safe to delete at any time;
the next run just re-fetches from UCI.

## Datasets used

| Dataset | UCI ID | Size | Source |
|---|---|---|---|
| [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) | 222 | 45,211 rows × 16 features | A Portuguese bank's term-deposit phone marketing campaign |
| [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) | 186 | 6,497 rows × 11 features | Physicochemical tests on Portuguese "Vinho Verde" wine samples |
| [Wholesale Customers](https://archive.ics.uci.edu/dataset/292/wholesale+customers) | 292 | 440 rows × 7 features | Annual spending by product category for a wholesale distributor's clients |
