dscompanion.tracking
=================

:func:`dscompanion.tracking_run` (documented in the :doc:`top-level API reference
<../api/dscompanion>`; implemented in ``dscompanion.tracking.run_context``) is a context manager
that logs a run's start and finish via Python's stdlib ``logging`` module — one INFO
line on entry, one on exit. It never fails and never blocks on a network call: there is
no external tracking server, no optional dependency, and no ``if tracking_available``
guard for calling code to write. Passing ``enabled=False`` skips logging entirely and
yields a no-op run handle with the same attribute shape.

The three functions below write additional structured INFO lines inside a
``tracking_run`` block, and are not re-exported at the top level:

- ``log_metrics`` — one INFO line per metric key/value (optionally tagged with a
  ``step`` for time-series logging, e.g. an epoch number).
- ``log_params`` — one INFO line for a dict of hyperparameters/config values; non-scalar
  values are coerced to ``str()`` before logging.
- ``log_artifact`` — records, via logging, that a local file was produced and its
  intended logical location. It does not copy or upload the file — the file already
  lives at its given local path.

There is no queryable run store. This module only ever writes to whatever log handler
the caller has configured — a caller that needs to browse past runs needs a different
mechanism (e.g. a log aggregator, or a purpose-built local run-store).

.. currentmodule:: dscompanion.tracking

.. autosummary::
   :toctree: ../api/generated
   :nosignatures:

   tracking_run
   log_metrics
   log_params
   log_artifact
