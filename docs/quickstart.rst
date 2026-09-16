Quickstart
==========

There are two ways to use dscompanion: the **YAML-driven pipeline** (recommended for anything
that will be reviewed, reproduced, or handed to another scientist), and the **low-level
building blocks** (useful for notebook exploration or when you need to deviate from the
fixed 13-stage pipeline order).

Recommended: YAML-driven pipeline
----------------------------------

Copy ``templates/experiment_template.yaml`` and fill in ``data.path`` / ``data.target`` —
everything else has a production-safe default. See :doc:`pipeline` for what each of the
13 stages does, and :doc:`configuration` for the full field reference.

.. code-block:: python

   from dscompanion.pipeline import PipelineRunner

   result = PipelineRunner.from_yaml("experiments/credit_risk_v1.yaml").run()

   print(result.metrics)                 # per-split metrics (train/val/test/oot)
   print(result.model)                   # fitted BaseDSCompanionModel subclass
   print(result.run_dir)                 # <reporting.output_dir>/<run_id>/ — this run's artifacts
   print(result.scoring_pipeline_path)   # ScoringPipeline bundle, auto-saved under run_dir/model/
   print(result.excel_report_path)       # Excel model card, auto-written under run_dir/reports/
   print(result.report_path)             # HTML model card (only when reporting.html_report=True)
   print(result.run_id)                  # local tracking run ID — matches result.run_dir.name

If ``leaderboard.enabled: true`` is set in the YAML, ``result.model`` is the **winning**
algorithm from the comparison — not necessarily ``model.algorithm`` as configured — and
``result.leaderboard`` holds the fitted :class:`~dscompanion.leaderboard.Leaderboard` instance
(its ``.leaderboard_`` attribute is the full ranked comparison table). See
:doc:`packages/leaderboard`.

Low-level: building blocks
---------------------------

Every stage is also usable directly, sklearn-style (``fit`` / ``transform`` /
``fit_transform``), if you need manual control over the sequence:

.. code-block:: python

   import dscompanion as ml
   import pandas as pd

   df = pd.read_parquet("data/credit_applications.parquet")

   # 1. Split — temporal, with an out-of-time holdout. snapshot_date holds a
   #    small number of discrete monthly values; the newest is always OOT,
   #    with train/val/test assigned automatically for 2-4 distinct values
   #    (see DataSplitter's docstring for the exact table, and
   #    date_value_roles for explicit control or 5+ distinct values).
   split = ml.DataSplitter(
       strategy="temporal",
       date_col="snapshot_date",
       target_col="default_flag",
   ).fit_split(df)

   # 2. EDA — univariate, bivariate, multivariate in one call
   eda = ml.EDAReport(split, target="default_flag").run_all()
   eda.to_html("outputs/eda.html")

   # 3. Feature processing — impute, encode, scale, leakage check
   fp = ml.FeatureProcessingPipeline(encoder="woe", scaler="robust")
   X_train_fp = fp.fit_transform(split.train_X, split.train_y)
   X_oot_fp = fp.transform(split.X_oot)

   # 4. Feature selection — constant, cardinality, correlation, IV
   fs = ml.FeatureSelectionPipeline()
   X_train_sel = fs.fit_transform(X_train_fp, split.train_y)
   X_oot_sel = fs.transform(X_oot_fp)

   # 5. Model — XGBoost with sensible defaults
   model = ml.ModelFactory.build(task="classification", algorithm="xgboost")
   model.fit(X_train_sel, split.train_y)
   print(model.evaluate(split))          # metrics across every available split

   # 6. Or compare every supported algorithm instead of picking one upfront
   leaderboard = ml.Leaderboard().run(split)
   model = leaderboard.best_model()

   # 7. Tune — Optuna trials, hyperparameter search over the model
   tuned_model = ml.Tuner(model=model, backend="optuna", n_trials=50, cv=split).run()

   # 8. Explain + calibrate
   explainer = ml.SHAPExplainer(tuned_model).fit(split.X_test)
   calibrator = ml.Calibrator(method="isotonic").fit(tuned_model, split.X_val, split.y_val)
   calibrated_model = calibrator.wrap(tuned_model)

   # 9. Document — governance-ready model card (HTML / Word / Excel / dict)
   card = ml.ModelCard(
       model=calibrated_model,
       split=split,
       explainer=explainer,
       calibrator=calibrator,
       template="credit_risk",
       author="Your Name",
       use_case="PD estimation for retail credit applications",
   ).generate()
   card.to_html("outputs/model_card.html")
   card.to_excel("outputs/model_card.xlsx")

Both paths produce the same kind of artefacts; the YAML path additionally gives you a
versioned, reviewable, re-runnable config file and wires every stage together with the
config-deviation reporting described in :doc:`pipeline`.

Scoring new data
-----------------

Training produces a ``ScoringPipeline`` bundle (``result.scoring_pipeline_path``) that
carries the fitted feature/selection pipelines and model together — the bare ``model``
alone expects already-preprocessed input and cannot be applied to raw data on its own.
Load the bundle in any later process and call ``predict`` on new, raw, unseen data:

.. code-block:: python

   from dscompanion.scoring import ScoringPipeline
   import pandas as pd

   scoring_pipeline = ScoringPipeline.load(result.scoring_pipeline_path)

   new_df = pd.read_parquet("data/credit_applications_2026_10.parquet")
   scored = scoring_pipeline.predict(new_df, id_columns=["application_id"])
   # columns: application_id, prediction, probability (classification only)

   # Single-record scoring — the natural binding for a future real-time API call
   one_result = scoring_pipeline.predict_one(new_df.iloc[0].to_dict())

   # Optional: check whether this batch's prediction distribution has drifted
   # from what training saw (Population Stability Index)
   drift_report = scoring_pipeline.compute_drift(new_df)

``predict`` validates ``new_df``'s schema against what training actually saw and raises a
clear ``ValueError`` naming any missing or dtype-incompatible column — never silently
produces predictions from a mismatched schema.

Batch scoring via YAML
------------------------

For a recurring batch scoring job (e.g. a scheduled Databricks notebook/job), copy
``templates/scoring_template.yaml`` instead of writing the Python above by hand — it gives you
the same versioned, reviewable, re-runnable config file the training side has, plus an
IST-timestamped, audited output run folder (scored file, a copy of the resolved config, a run
log, and an optional drift report), mirroring :class:`~dscompanion.PipelineRunner`'s run-folder
convention:

.. code-block:: python

   from dscompanion.scoring import ScoringRunner

   result = ScoringRunner.from_yaml("scoring/credit_risk_v1_scoring.yaml").run()

   print(result.scored_df)          # same shape as ScoringPipeline.predict()'s return value
   print(result.output_path)        # <output.output_dir>/<run_id>/scored.parquet
   print(result.drift_report)       # populated only when check_drift: true in the YAML
