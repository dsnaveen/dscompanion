# dscompanion

A production-grade ML pipeline toolkit for tabular data: EDA, feature engineering, model
training/tuning/calibration, and governance-ready model cards — with sensible defaults
everywhere and full override capability. Designed to work well in regulated or restricted
environments (see [Architecture](docs/architecture.rst) for the constraints that shaped it),
but useful for any tabular ML project.

## Installation

```bash
pip install dscompanion
```

For local development:

```bash
git clone https://github.com/dsnaveen/dscompanion.git
cd dscompanion
pip install -e ".[dev]"
```

**One thing worth knowing about SHAP:** `ExplainConfig.shap_enabled` defaults to `False` —
SHAP computation is relatively expensive, so it's opt-in rather than run automatically. See
`explain.permutation_enabled` for a lighter-weight, model-agnostic alternative.

**Run tracking via stdlib logging** — dscompanion logs pipeline execution (run start/finish,
training metrics, parameters, artifacts) via Python's `logging` module, not a dedicated
tracking server. A `dscompanion.tracking.run_context.tracking_run()` context manager handles
this transparently during pipeline execution. If you need queryable run history or a tracking
server UI, you'll need a separate mechanism outside dscompanion — it provides only local
structured logs by default.

## Docker

A self-contained image with every optional extra pre-installed is published to
GitHub Container Registry on every release:

```bash
docker pull ghcr.io/dsnaveen/dscompanion:latest
docker run -it ghcr.io/dsnaveen/dscompanion:latest
```

Also works with [Podman](https://podman.io/) — `podman pull`/`podman run` are
drop-in replacements for the `docker` commands above.

The default command drops you into a shell with dscompanion and every optional
dependency already installed. To run the optional FastAPI service instead, override
the command:

```bash
docker run -p 8000:8000 ghcr.io/dsnaveen/dscompanion:latest \
  uvicorn dscompanion.api.main:app --host 0.0.0.0
```

## Quickstart

```python
from dscompanion.pipeline import PipelineConfig, PipelineRunner

cfg = PipelineConfig(
    name="my_first_model",
    data={"path": "data.parquet", "target": "target_column"},
    split={"method": "stratified", "test_size": 0.2, "val_size": 0.1},
    model={"task": "classification", "algorithm": "xgboost"},
    reporting={"output_dir": "outputs", "html_report": True},
)
result = PipelineRunner(cfg).run()

print(result.metrics)                 # per-split evaluation metrics
print(result.model_card.to_excel(...))  # governance-ready model card
```

`PipelineRunner` runs the full pipeline end-to-end — load, split, EDA, feature processing,
selection, imbalance handling, train (or leaderboard comparison), tune, calibrate, explain,
and report — with every stage overridable through `PipelineConfig`. See
[`docs/pipeline.rst`](docs/pipeline.rst) for the full stage-by-stage reference and
[`docs/quickstart.rst`](docs/quickstart.rst) for more examples.

## Package Structure

```
src/dscompanion/
├── config.py                  # DSCompanionConfig settings (pydantic-settings, env_prefix=DSCOMPANION_)
├── pipeline/                  # PipelineConfig, PipelineRunner — the end-to-end orchestrator
├── split/                     # DataSplitter — temporal, stratified, grouped, random
├── eda/                       # UnivariateAnalyser, BivariateAnalyser, MultivariateAnalyser, EDAReport
├── features/                  # SmartImputer, encoders, SmartScaler, AutoBinner,
│                              #   DateFeatureExtractor, LeakageGuard, FeatureProcessingPipeline
├── targets/                   # ImbalanceHandler (SMOTE/class_weight), TargetBinariser
├── selection/                 # ConstantSelector, CorrelationSelector, IVSelector,
│                              #   FeatureSelectionPipeline
├── models/                    # ModelFactory, ClassificationModel, RegressionModel, ClusteringModel
├── tuning/                    # Tuner, OptunaBackend, HyperoptBackend, search_spaces
├── leaderboard/                # Leaderboard — multi-algorithm comparison
├── explain/                   # SHAPExplainer, PermutationImportanceAnalyser
├── calibration/                # Calibrator (isotonic, Platt, beta)
├── docs/                      # ModelCard, HTML report generators (EDA/model cards, widgets)
├── tracking/                   # tracking_run context manager, log_metrics/params/artifact via stdlib logging
├── api/                       # optional FastAPI service (pip install dscompanion[api])
└── utils/                     # metrics (KS, Gini, PSI, IV, WoE, ECE), validators, plotting
```

A Streamlit UI lives at [`app/`](app/) (`pip install dscompanion[app]`, then
`streamlit run app/streamlit_app.py`) for interactively driving a pipeline run without
writing config by hand.

## Configuration

All thresholds and defaults are controlled via `dscompanion.config.settings`. Override with environment
variables prefixed `DSCOMPANION_`:

```bash
export DSCOMPANION_TARGET_LEAKAGE_CORRELATION_THRESHOLD=0.90
export DSCOMPANION_HIGH_CARDINALITY_THRESHOLD=30
export DSCOMPANION_RANDOM_STATE=0
```

Or in code:

```python
from dscompanion.config import settings
settings.high_cardinality_threshold = 30
```

## Running Tests

```bash
git clone https://github.com/dsnaveen/dscompanion.git
cd dscompanion
pip install -e ".[dev]"
pytest tests/ -v
```

## Key Design Principles

- **sklearn-compatible** — all transformers implement `fit` / `transform` / `get_feature_names_out`
- **joblib-serialisable** — every fitted object can be pickled with `joblib.dump`
- **Structured logging via stdlib** — all pipeline execution logs through Python's `logging` module
- **pydantic v2** — config and report models use `BaseModel` / `BaseSettings`
- **No hidden network calls** — works fully offline once installed, suited to restricted environments

## Dependencies

Core: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `pydantic-settings`, `joblib`, `plotly`, `jinja2`

Optional: `optuna`, `shap`, `python-docx` (core extras); `streamlit` (`[app]`); `fastapi`,
`uvicorn` (`[api]`); `sphinx` (`[docs]`)

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT — see [`LICENSE`](LICENSE).
