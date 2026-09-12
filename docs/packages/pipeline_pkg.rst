dscompanion.pipeline
=================

The orchestration layer. :class:`dscompanion.PipelineConfig` is the pydantic schema (see
:doc:`../configuration` for how it differs from ``dscompanion.config.settings``), and
:class:`dscompanion.PipelineRunner` executes the 13 fixed stages described in
:doc:`../pipeline` against it, returning a :class:`dscompanion.PipelineRunResult` with every
fitted artefact. All three are re-exported at the top level and documented in the
:doc:`top-level API reference <../api/dscompanion>`.

The nested config sections below (each a field on ``PipelineConfig``) are not
re-exported at the top level — use e.g. ``dscompanion.pipeline.ModelConfig``, or just write
the equivalent YAML key (see ``templates/experiment_template.yaml`` for the exhaustive,
load-tested reference).

.. currentmodule:: dscompanion.pipeline

.. autosummary::
   :toctree: ../api/generated
   :nosignatures:

   DataConfig
   SplitConfig
   EDAConfig
   FeaturesConfig
   ImputerConfig
   EncoderConfig
   ScalerConfig
   TargetConfig
   ImbalanceConfig
   SelectionConfig
   ModelConfig
   TuningConfig
   LeaderboardConfig
   ExplainConfig
   ReportingConfig
