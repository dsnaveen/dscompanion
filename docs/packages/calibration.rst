dscompanion.calibration
====================

:class:`dscompanion.Calibrator` is documented in the :doc:`top-level API reference
<../api/dscompanion>` (it's the sub-package's only public class, and is re-exported at the
top level).

``Calibrator(method="isotonic", cv="prefit")`` wraps probability calibration —
isotonic, Platt, or beta — over an already-fitted classifier. ``.fit(model, X_cal,
y_cal)`` learns the calibration mapping; ``.wrap(model)`` returns a wrapper object whose
``predict_proba`` applies that mapping, without mutating the original model. Pipeline
stage 11 (see :doc:`../pipeline`) calls this with method/cv hardcoded — there is
currently no ``PipelineConfig`` section for choosing a different method or cv strategy
per experiment.
