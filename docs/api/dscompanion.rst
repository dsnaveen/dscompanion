Top-level API (``import dscompanion as ml``)
==========================================

Every name below is re-exported at the package root — i.e. reachable as ``ml.<Name>``
after ``import dscompanion as ml``, without needing to know which sub-package it lives in.
For the full per-sub-package reference (including classes not re-exported at the top
level, like the individual feature selectors), see the :doc:`../index` sub-package
pages.

.. currentmodule:: dscompanion

.. autosummary::
   :toctree: generated
   :nosignatures:

   DataSplitter
   DataSplit
   EDAReport
   FeatureProcessingPipeline
   ImbalanceHandler
   TargetBinariser
   FeatureSelectionPipeline
   ModelFactory
   Leaderboard
   Tuner
   SHAPExplainer
   BootstrapSHAPExplainer
   LIMEExplainer
   Calibrator
   ModelCard
   tracking_run
   PipelineConfig
   PipelineRunner
   PipelineRunResult
   ScoringPipeline
   SyntheticDataGenerator

``ml.settings`` is also re-exported, but it's a singleton *instance* of
``dscompanion.config.Settings`` (not a class), so it's documented narratively in
:doc:`../configuration` instead of here.
