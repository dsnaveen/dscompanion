dscompanion.utils
=============

:class:`dscompanion.SyntheticDataGenerator` (documented in the :doc:`top-level API reference
<../api/dscompanion>`) produces datasets with controlled data quality characteristics
(high-null, low-variance, constant, high-cardinality columns) — used by dscompanion's own
test suite and for team training sessions.

The functions below — classification/ranking metrics (KS, Gini, PSI, IV, WoE, Expected
Calibration Error), input validators (DataFrame type/shape checks, binary-target
checks, a Databricks environment probe), and Plotly theming helpers — are used
throughout the rest of the package but are not re-exported at the top level.

.. currentmodule:: dscompanion.utils

.. autosummary::
   :toctree: ../api/generated
   :nosignatures:

   ks_statistic
   gini_coefficient
   psi_score
   iv_score
   woe_bins
   expected_calibration_error
   validate_dataframe
   validate_binary_target
   is_databricks
   apply_dscompanion_theme
   fig_to_base64
