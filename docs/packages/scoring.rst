dscompanion.scoring
====================

Every public class here — :class:`dscompanion.ScoringPipeline`, :class:`dscompanion.ScoringConfig`,
:class:`dscompanion.ScoringRunner`, :class:`dscompanion.ScoringRunResult` — is documented in
the :doc:`top-level API reference <../api/dscompanion>` (all four are re-exported at the top
level; ``ScoringDataConfig``/``ScoringOutputConfig``, ``ScoringConfig``'s nested sections, are
not — see ``ScoringConfig`` itself for their fields).

``ScoringPipeline`` bundles a trained model with its fitted preprocessing chain
(``feature_pipeline``, ``selection_pipeline``, and optionally a ``calibrator``) so it can
be correctly applied to new, raw, unseen data in a later process — see :doc:`../quickstart`'s
"Scoring new data" section for usage. It never fits anything itself: ``PipelineRunner``
builds one from a completed run's fitted stage artifacts (``ScoringPipeline.from_run()``)
and saves it automatically as ``<run_dir>/model/<name>_v<version>_scoring_pipeline.joblib``
(``PipelineRunResult.scoring_pipeline_path``). ``.predict(df, id_columns=...)`` scores a
batch and returns a DataFrame; ``.predict_one(record)`` scores a single dict record — the
natural binding for a future single-request API endpoint; ``.compute_drift(df)`` checks the
batch's prediction-score distribution against the training-time reference via Population
Stability Index; ``.compute_feature_drift(df)`` checks per-raw-feature drift (Characteristic
Stability Index) against a frozen, privacy-safe training reference distribution (bin
edges/proportions, never raw rows) — see :doc:`monitoring` for how this feeds a full
monitoring report once actuals are available.

``ScoringConfig``/``ScoringRunner`` are the YAML-driven counterpart to
``PipelineConfig``/``PipelineRunner``, for a recurring batch scoring job — see
:doc:`../quickstart`'s "Batch scoring via YAML" section and ``templates/scoring_template.yaml``.
``ScoringRunner.from_yaml(path).run()`` loads the input data and a ``ScoringPipeline`` bundle,
scores it, and writes the result into a timestamped ``<output.output_dir>/<run_id>/`` folder —
the same IST-timestamped, audited run-folder convention ``PipelineRunner`` uses for training.
