dscompanion.scoring
====================

:class:`dscompanion.ScoringPipeline` is documented in the :doc:`top-level API reference
<../api/dscompanion>` (it's the sub-package's only public class, and is re-exported at the
top level).

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
Stability Index.
