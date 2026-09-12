dscompanion.eda
==========

:class:`dscompanion.EDAReport` (documented in the :doc:`top-level API reference
<../api/dscompanion>`) is a facade — ``EDAReport.run_all()`` delegates to the four analysers
below: ``UnivariateAnalyser`` (per-column stats, distributions), ``BivariateAnalyser``
(IV/WoE, event rate vs. target), ``MultivariateAnalyser`` (correlation matrix, PCA), and
``MissingnessAnalyser`` (nullity matrix, missingness correlation). ``EDAReport`` is also
what :class:`dscompanion.ModelCard` reads from when an ``eda_report`` is supplied.

None of the four analysers below are re-exported at the top level — use
``dscompanion.eda.UnivariateAnalyser`` etc., or call them indirectly via ``EDAReport``.

.. currentmodule:: dscompanion.eda

.. autosummary::
   :toctree: ../api/generated
   :nosignatures:

   UnivariateAnalyser
   BivariateAnalyser
   MultivariateAnalyser
   MissingnessAnalyser
