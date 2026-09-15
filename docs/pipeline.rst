The 13-Stage Pipeline
======================

:class:`~dscompanion.pipeline.PipelineRunner` runs these 13 stages in fixed order — they can
be configured (a step can do more or less work, or be skipped) but never reordered.
Stage numbers below match the ``[N/13]`` log lines emitted at INFO level during a run.

.. list-table::
   :header-rows: 1
   :widths: 8 25 67

   * - #
     - Stage
     - What happens / how to configure it
   * - 1
     - Loading data
     - Reads ``data.path`` (parquet/csv/excel/delta) via ``DataConfig``. ``data.nrows`` caps
       row count for dev iteration.
   * - 2
     - Splitting data
     - ``DataSplitter`` builds the train/val/test/OOT partitions per ``split.method``
       (stratified / random / temporal / grouped). ``test``/``train`` are always
       non-empty; ``val``/``oot`` may be empty depending on config.
   * - 3
     - EDA
     - ``EDAReport.run_all()`` — univariate, bivariate, multivariate, missingness.
       Controlled by ``eda.*`` (see :doc:`configuration`). Skipped entirely if
       ``eda.enabled=False``.
   * - 4
     - Target treatment
     - Optional continuous→binary conversion via ``target.binarize_threshold``.
   * - 5
     - Feature processing
     - Impute → encode → scale, via ``FeatureProcessingPipeline``, configured by
       ``features.imputer`` / ``features.encoder`` / ``features.scaler`` /
       ``features.winsorizer``. Includes the leakage check.
   * - 6
     - Feature selection
     - The fixed selector chain (null-rate → constant → quasi-constant → cardinality →
       leakage → correlation → IV → VIF) via ``FeatureSelectionPipeline``, configured by
       ``selection.*``.
   * - 7
     - Imbalance handling
     - ``ImbalanceHandler`` applies ``target.imbalance.strategy`` (class_weight / SMOTE
       / under/oversample / none) — classification only.
   * - 8
     - Training **or** Leaderboard
     - **Default**: builds and fits exactly ``model.algorithm`` via ``ModelFactory``.
       **If** ``leaderboard.enabled=True``: every algorithm supported for
       ``model.task`` (or the ``include``/``exclude`` subset) is trained and ranked by
       :class:`~dscompanion.leaderboard.Leaderboard` instead, and the winner replaces
       ``model.algorithm`` *in place* for every subsequent stage. See
       :doc:`packages/leaderboard`.
   * - 9
     - Hyperparameter tuning (optional)
     - ``Tuner`` runs ``tuning.n_trials`` Optuna trials against ``tuning.metric`` when
       ``tuning.enabled=True``; skipped otherwise.
   * - 10
     - Evaluating
     - ``model.evaluate(split)`` — metrics on every available split (train/val/test/oot),
       a tidy long-format table.
   * - 11
     - Calibration
     - Classification only. ``Calibrator(method="isotonic", cv="prefit")`` fit on the
       validation split — hardcoded in ``PipelineRunner._calibrate()``, no YAML
       override exists for this stage today.
   * - 12
     - Explainability (optional)
     - ``SHAPExplainer`` on a ``explain.shap_sample_size``-row sample of the test split,
       when ``explain.shap_enabled=True``. Independently, ``PermutationImportanceAnalyser``
       on a ``explain.permutation_sample_size``-row sample, when
       ``explain.permutation_enabled=True`` — model-agnostic and generally
       lighter-weight than SHAP. Either, both, or neither may run.
   * - 13
     - Model card + run logging
     - ``ModelCard.generate()`` assembles every prior stage's output into one report.
       A single ``reporting.output_dir`` is declared once; ``PipelineRunner`` organizes
       everything else itself under ``<output_dir>/<run_id>/`` — ``model/`` (the trained
       model, auto-saved via ``model.save()``), ``reports/`` (Excel model card, always
       written; HTML, only when ``reporting.html_report=True``), ``logs/`` (this run's
       captured log lines), and ``eda/`` (reserved). The resolved config is also saved as
       ``<run_dir>/config.yaml``. Run tracking via
       ``dscompanion.tracking.run_context.tracking_run()`` (stdlib ``logging``, not MLflow) —
       see :doc:`packages/tracking`.

Config-deviation reporting
----------------------------

Before stage 1 runs, ``PipelineRunner._detect_config_deviations()`` compares every
field in ``PipelineRunner._config_checks_spec()`` against its documented default and
logs a ``⚠`` warning for each one that differs — e.g. ``tuning.enabled=True`` when the
default is ``False``. The same comparison, **including every non-deviating field**,
becomes the "Configuration Summary" table at the top of the model card (stage 13), so a
reviewer can confirm "no surprises" by reading one table instead of the full YAML.

This list is the single source of truth for what counts as a governance-relevant
choice — identity fields (``name``, ``version``, ``owner``), ``data.*``, and ``split.*``
are deliberately excluded (experiment-specific, not a "default vs override" choice),
as are routine numeric thresholds like ``iv_threshold``.

What happens when leaderboard mode is on
-------------------------------------------

Stage 8's branch is the one place where the pipeline's behaviour structurally changes
based on config, rather than just skipping work. With ``leaderboard.enabled=True``:

1. ``model.algorithm`` is read by ``PipelineConfig`` validation (so it must still be a
   real algorithm name) but is **ignored** at runtime.
2. :class:`~dscompanion.leaderboard.Leaderboard` trains every candidate once on
   ``split.X_train``/``split.y_train`` and evaluates once on ``val`` (falling back to
   ``test``) — never on OOT, to preserve it as an unbiased final holdout.
3. The top-ranked candidate's algorithm name overwrites ``config.model.algorithm`` in
   place, so stages 9–13 (tuning, evaluation, calibration, SHAP, model card, run
   logging) all see and report the algorithm that was *actually* trained.
4. ``PipelineRunResult.leaderboard`` holds the fitted ``Leaderboard`` instance —
   ``result.leaderboard.leaderboard_`` is the full ranked comparison table,
   ``result.leaderboard.fitted_models_`` holds every successfully-trained candidate,
   not just the winner.
