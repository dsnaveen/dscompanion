dscompanion.selection
=================

:class:`dscompanion.FeatureSelectionPipeline` (documented in the :doc:`top-level API
reference <../api/dscompanion>`) runs the chain below in fixed order for pipeline stage 6
(see :doc:`../pipeline`): ``NullRateSelector`` → ``ConstantSelector`` →
(quasi-constant, via ``ConstantSelector``'s threshold) → ``CardinalitySelector`` →
leakage check → ``CorrelationSelector`` → ``IVSelector`` → VIF (optional).
``RFESelector`` and ``SHAPSelector`` are available but not part of the default chain —
opt in via ``FeatureSelectionPipeline(selectors=[...])``.

None of the selector classes below are re-exported at the top level — use e.g.
``dscompanion.selection.IVSelector``.

.. currentmodule:: dscompanion.selection

.. autosummary::
   :toctree: ../api/generated
   :nosignatures:

   BaseSelector
   NullRateSelector
   ConstantSelector
   CardinalitySelector
   CorrelationSelector
   IVSelector
   RFESelector
   SHAPSelector
