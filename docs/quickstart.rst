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
   print(result.model_path)              # trained model, auto-saved under run_dir/model/
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

   # 1. Split — temporal, with an out-of-time holdout
   split = ml.DataSplitter(
       strategy="temporal",
       date_col="snapshot_date",
       target_col="default_flag",
       oot_cutoff="2023-09-01",
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
