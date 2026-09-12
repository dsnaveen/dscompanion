Architecture
============

Why dscompanion exists
------------------

dscompanion builds classification/regression/clustering models for
regulated or otherwise restricted environments, designed around three
common hard constraints:

1. **Restricted or no external internet access for packages** — new Python
   packages can require a lengthy approval process, so the toolkit works
   entirely with what's already installed rather than pulling in new
   dependencies on a whim.
2. **Compliance/risk/audit review of every model** — every config choice, every
   non-default override, and every metric needs to be traceable and explainable to a
   reviewer who did not write the code.
3. **Limited or no direct terminal access to the production environment
   during development** — code is often written locally and only later
   deployed to a managed platform without a debugger attached.

dscompanion answers this with a fixed, ordered pipeline (see :doc:`pipeline`) where every
stage has a safe default, every default is overridable through one validated config
object, and every run produces a self-contained governance artefact (the model card)
documenting exactly what ran and what deviated from the defaults.

Sub-package map
----------------

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Package
     - Responsibility
   * - :doc:`packages/split`
     - ``DataSplitter`` — temporal / stratified / grouped / random train-val-test-OOT splits.
   * - :doc:`packages/eda`
     - ``EDAReport`` and its four analysers (univariate, bivariate, multivariate,
       missingness) — descriptive statistics, IV/WoE, correlation, alerts.
   * - :doc:`packages/features`
     - ``SmartImputer``, encoders (ordinal/one-hot/WoE/target), ``SmartScaler``,
       ``AutoBinner``, ``WinsorizationTransformer``, ``LeakageGuard``, and the
       ``FeatureProcessingPipeline`` that chains them.
   * - :doc:`packages/targets`
     - ``TargetBinariser`` (continuous → binary) and ``ImbalanceHandler``
       (class_weight / SMOTE / under/oversampling).
   * - :doc:`packages/selection`
     - The selector chain (null-rate, constant, quasi-constant, cardinality, leakage,
       correlation, IV, VIF) and ``FeatureSelectionPipeline``.
   * - :doc:`packages/models`
     - ``ModelFactory`` plus the three task wrappers — ``ClassificationModel``,
       ``RegressionModel``, ``ClusteringModel`` — all subclassing ``BaseDSCompanionModel``.
   * - :doc:`packages/leaderboard`
     - ``Leaderboard`` — trains and ranks every supported classification algorithm on
       one ``DataSplit``, PyCaret/H2O AutoML-style.
   * - :doc:`packages/tuning`
     - ``Tuner`` plus Optuna/Hyperopt backends and the per-algorithm ``SEARCH_SPACES``.
   * - :doc:`packages/calibration`
     - ``Calibrator`` — isotonic / Platt / beta probability calibration.
   * - :doc:`packages/explain`
     - ``SHAPExplainer``, ``BootstrapSHAPExplainer``, ``LIMEExplainer``.
   * - :doc:`packages/docs`
     - ``ModelCard`` — the governance report (HTML / Word / Excel / dict), built from
       every other stage's output.
   * - :doc:`packages/tracking`
     - ``tracking_run`` context manager plus ``log_metrics``/``log_params``/``log_artifact`` —
       local run tracking via stdlib ``logging``, no external server dependency.
   * - :doc:`packages/pipeline_pkg`
     - ``PipelineConfig`` (the pydantic schema) and ``PipelineRunner`` (the
       YAML-to-model-card orchestrator) — ties every other sub-package together.
   * - :doc:`packages/utils`
     - Metrics (KS, Gini, PSI, IV, WoE, ECE, decile table), input validators, plotting
       helpers, and ``SyntheticDataGenerator`` for tests/training.

Conventions that hold across every sub-package
------------------------------------------------

- **sklearn-compatible transformers.** Every feature/selection transformer implements
  ``fit`` / ``transform`` / ``fit_transform`` / ``get_params`` / ``set_params``, so it
  drops into an sklearn ``Pipeline`` if needed.
- **pydantic v2 everywhere config matters.** ``dscompanion.config.settings`` (a
  ``pydantic-settings`` ``BaseSettings`` singleton) holds environment-wide numeric
  defaults; ``dscompanion.pipeline.config.PipelineConfig`` (a ``pydantic`` ``BaseModel`` with
  ``extra="forbid"``) is the per-experiment schema loaded from YAML. See
  :doc:`configuration` for the distinction — conflating the two is the single most
  common config mistake.
- **joblib-serialisable.** Every fitted model/explainer can be ``joblib.dump``'d; its
  path can be recorded in the run log via ``dscompanion.tracking.run_context.log_artifact()``.
- **stdlib logging only.** Every module uses ``logging.getLogger(__name__)`` with
  ``%``-style format strings — never f-strings in log calls, never ``print()``. No
  customer-level data is ever logged (row counts and column names only).
- **Run tracking never fails and never blocks.** ``tracking_run()`` and
  ``log_metrics``/``log_params``/``log_artifact`` write to stdlib logging only — no
  network call, no external tracking server, and no optional dependency for callers to
  guard against.
- **No Spark, no Databricks magics, no ``display()``/``dbutils`` calls inside the
  library.** dscompanion is pandas-only; Spark↔pandas conversion is the caller's
  responsibility, so the same code runs identically in a notebook cell or a packaged
  ``.py`` job.
