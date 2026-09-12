Configuration
=============

dscompanion has **two separate config systems**. Confusing them is the single most common
mistake when writing an experiment YAML — both are legitimate, they just answer
different questions.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * -
     - ``dscompanion.config.settings``
     - ``dscompanion.pipeline.config.PipelineConfig``
   * - Scope
     - Whole environment / team — set once
     - One experiment run — set per YAML
   * - Loaded by
     - Imported automatically the moment any dscompanion module imports ``dscompanion.config``
     - ``PipelineRunner.from_yaml(path)``, explicitly, per run
   * - Example fields
     - ``vif_round_precision``, ``score_dist_bins``, ``splitter_oot_min_rows``
     - ``data.path``, ``model.algorithm``, ``leaderboard.enabled``
   * - Changed via
     - Python: ``settings.vif_round_precision = 3``
     - The experiment YAML file itself

``PipelineConfig`` is a ``pydantic`` ``BaseModel`` with ``extra="forbid"`` on every
nested section — any YAML key that isn't an explicitly declared field raises a
``pydantic.ValidationError`` *before a single pipeline stage runs*. This is deliberate:
it is what lets a reviewer trust that "the YAML matched what ran." It also means a
``Settings``-only field pasted into the YAML by mistake fails loudly and immediately,
not silently.

The canonical, exhaustive, **load-tested** reference for every real
``PipelineConfig`` field is ``templates/experiment_template.yaml`` at the project root —
it is kept in sync with the schema (verified via ``PipelineConfig.from_yaml()`` +
``to_yaml()`` round-trip), with a clearly separated comment block at the bottom
documenting every ``Settings``-only field that has *no* per-experiment override.

``Settings`` field categories
-------------------------------

Every ``Settings`` field is read via ``settings.<field>`` somewhere in library code —
never hardcoded. Roughly:

- **Splitting** — ``splitter_oot_min_rows``, ``splitter_class_ratio_tolerance``.
- **EDA internals** — chart bins, top-N counts, alert thresholds not exposed via
  ``EDAConfig`` (PCA component counts, IV bin counts, output rounding precision).
- **Feature engineering internals** — ``AutoBinner`` defaults, WoE clipping/smoothing,
  imputer auto-strategy skew threshold — none of these have a ``PipelineConfig``
  override; ``features.binner`` is **not** a real YAML section.
- **Model/metric internals** — ``score_dist_bins``, ``evaluate_round_precision``,
  ``classification_metrics`` / ``regression_metrics`` / ``clustering_metrics`` (the
  latter is also :class:`~dscompanion.leaderboard.Leaderboard`'s default sort-metric source).
- **Explainability internals** — SHAP background-sample counts, bootstrap batch sizing.
- **Calibration** — ``calibration_ece_bins`` only; the calibration *method* (isotonic)
  and *cv strategy* (prefit) are hardcoded in ``PipelineRunner._calibrate()`` with no
  config surface at all today.
- **Selection internals not in** ``SelectionConfig`` — ``near_zero_variance_threshold``,
  plus a few fields that duplicate a ``SelectionConfig`` field by a different name
  (e.g. ``correlation_threshold`` exists in both places — the ``Settings`` one has no
  effect once ``selection.correlation_threshold`` is set, since selectors are built from
  the ``PipelineConfig`` value).

When in doubt about whether a field belongs in the YAML or in ``settings``: if it
appears as a documented field in one of the ``pydantic`` models in
``dscompanion/pipeline/config.py`` (see :doc:`packages/pipeline_pkg`), it's YAML. Otherwise
it's ``settings``.
