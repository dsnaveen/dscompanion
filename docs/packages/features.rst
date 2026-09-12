dscompanion.features
================

:class:`dscompanion.FeatureProcessingPipeline` (documented in the :doc:`top-level API
reference <../api/dscompanion>`) chains the transformers below together for pipeline stage 5
(see :doc:`../pipeline`): imputation, encoding, scaling, binning, date-feature
extraction, winsorization, and leakage detection. ``LeakageGuard`` runs inside that
pipeline and raises ``LeakageError`` (or returns a ``LeakageReport``, depending on
strictness) when a feature is suspiciously well-correlated with the target.

None of the classes below are re-exported at the top level — use e.g.
``dscompanion.features.SmartImputer``.

.. currentmodule:: dscompanion.features

.. autosummary::
   :toctree: ../api/generated
   :nosignatures:

   SmartImputer
   OrdinalEncoder
   OneHotEncoder
   WoEEncoder
   HighCardinalityEncoder
   SmartScaler
   WinsorizationTransformer
   AutoBinner
   DateFeatureExtractor
   LeakageGuard
   LeakageReport
   LeakageError

Standalone transformers
------------------------

Defined, tested, and importable from ``dscompanion.features``, but **not** wired into
:class:`dscompanion.FeatureProcessingPipeline`'s automatic column-detection stage — each is an
optional, model-specific choice you opt into explicitly in a pipeline recipe, rather than a
universal stage every dataset runs through. Several need more than a single column at a
time (a group-by key, several columns compressed into few, cross-column imputation, two
named columns combined) and so cannot be a per-column recipe step in the first place.

.. currentmodule:: dscompanion.features

.. autosummary::
   :toctree: ../api/generated
   :nosignatures:

   BinaryEncoder
   CategoryCombinerTransformer
   ColumnArithmeticTransformer
   DistributionTransformer
   GroupRelativeTransformer
   HashEncoder
   MultivariateImputer
   NoiseInjector
   PCATransformer
   PercentileRankTransformer
   PolynomialFeaturesTransformer
   RareCategoryGrouper
   SplineFeatureTransformer
   StatisticalOutlierCapper
