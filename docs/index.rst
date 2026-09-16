dscompanion documentation
=====================

**dscompanion** is an internal ML acceleration package for banking analytics —
a production-grade toolkit covering the full lifecycle of a credit-risk / marketing
classification model: data splitting, EDA, feature engineering, feature selection,
model training (including a PyCaret/H2O-style algorithm leaderboard), hyperparameter
tuning, calibration, explainability, and a governance-ready model card.

It is designed around a "first model in under a day" principle: every stage has a
production-safe default, every default is overridable, and the entire pipeline can be
driven end-to-end from a single YAML file via :class:`~dscompanion.pipeline.PipelineRunner`.

.. toctree::
   :maxdepth: 2
   :caption: Guide

   quickstart
   architecture
   pipeline
   configuration

.. toctree::
   :maxdepth: 1
   :caption: Sub-packages

   packages/split
   packages/eda
   packages/features
   packages/targets
   packages/selection
   packages/models
   packages/leaderboard
   packages/tuning
   packages/calibration
   packages/explain
   packages/docs
   packages/tracking
   packages/pipeline_pkg
   packages/scoring
   packages/utils

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/dscompanion

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
