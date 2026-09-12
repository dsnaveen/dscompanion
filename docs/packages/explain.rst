dscompanion.explain
================

Model explainability. :class:`dscompanion.SHAPExplainer` auto-selects ``TreeExplainer``
(tree models) or ``LinearExplainer``/``KernelExplainer`` otherwise;
:class:`dscompanion.BootstrapSHAPExplainer` repeats SHAP computation over bootstrap batches
for confidence-interval-style importance estimates on very large datasets.
:class:`dscompanion.LIMEExplainer` gives local, per-instance explanations. All three are
re-exported at the top level and documented in the :doc:`top-level API reference
<../api/dscompanion>`. Pipeline stage 12 (see :doc:`../pipeline`) uses ``SHAPExplainer`` and/or
``PermutationImportanceAnalyser``, each independently gated by its own config flag, on a
sample of the test split.

``PDPAnalyser`` (below) computes partial dependence plots and is not re-exported at the
top level. ``PermutationImportanceAnalyser`` computes feature importance via
``sklearn.inspection.permutation_importance`` against an already-fitted estimator — unlike
``SHAPExplainer``, it needs no ``shap.TreeExplainer`` at all, so it's model-agnostic and
generally lighter-weight. Pipeline stage 12 uses it independently of
``SHAPExplainer`` (``explain.permutation_enabled``, separate from ``explain.shap_enabled``) —
either, both, or neither may be on for a given run. Also not re-exported at the top level.

.. currentmodule:: dscompanion.explain

.. autosummary::
   :toctree: ../api/generated
   :nosignatures:

   PDPAnalyser
   PermutationImportanceAnalyser
