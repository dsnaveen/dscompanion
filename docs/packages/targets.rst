dscompanion.targets
===============

Target-variable treatment, run in pipeline stages 4 and 7 (see :doc:`../pipeline`):
:class:`dscompanion.TargetBinariser` converts a continuous target to binary at a threshold;
:class:`dscompanion.ImbalanceHandler` applies class-weighting, SMOTE, or under/oversampling
for classification tasks.

Both classes are re-exported at the top level and documented in the
:doc:`top-level API reference <../api/dscompanion>` — there is nothing sub-package-exclusive
to document separately here.
