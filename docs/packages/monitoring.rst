dscompanion.monitoring
========================

Every public class here — :class:`dscompanion.MonitoringConfig`,
:class:`dscompanion.MonitoringRunner`, :class:`dscompanion.MonitoringRunResult` — is
documented in the :doc:`top-level API reference <../api/dscompanion>` (all three are
re-exported at the top level; ``MonitoringActualsConfig``/``MonitoringOutputConfig``,
``MonitoringConfig``'s nested sections, are not — see ``MonitoringConfig`` itself for
their fields).

``MonitoringConfig``/``MonitoringRunner`` are the third pillar of the scorecard lifecycle,
after training (:doc:`pipeline_pkg`) and scoring (:doc:`scoring`) — see :doc:`../quickstart`'s
"Monitoring" section and ``templates/monitoring_template.yaml``. A monitoring run joins a
past :class:`~dscompanion.ScoringRunner` output file (``scored_data``) with a
separately-arrived actuals file (``actuals_data``) by ``id_columns``, re-measures
performance against the actual outcome using the exact same metric formulas
``BaseDSCompanionModel.evaluate()`` uses, and — when ``raw_data`` is supplied — computes
feature-level drift (CSI) against the training reference via
:meth:`~dscompanion.ScoringPipeline.compute_feature_drift`.

Deliberately evaluates what was *actually* predicted at scoring time (possibly by an
older model version), not a re-prediction with the current model — this is why
``id_columns`` must match what the original ``ScoringRunner`` run used: it's the only
join key available between the scored output and the actuals that arrived later.

``MonitoringRunner.from_yaml(path).run()`` writes the result into a timestamped
``<output.output_dir>/<run_id>/`` folder — the same IST-timestamped, audited run-folder
convention ``PipelineRunner``/``ScoringRunner`` use for training/scoring.

This is a single point-in-time snapshot report per run — there is no persistent
longitudinal history store for tracking metrics across many monitoring runs over time
(tracked separately as a future enhancement).
