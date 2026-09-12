dscompanion.leaderboard
===================

:class:`dscompanion.Leaderboard` is documented in the :doc:`top-level API reference
<../api/dscompanion>` (it's the sub-package's only public class, and is re-exported at the
top level).

PyCaret's ``compare_models()`` / H2O AutoML's leaderboard, adapted to dscompanion's existing
``Tuner``-style constructor-config + ``run(split)`` shape. Classification only for now.
Trains every algorithm in ``ModelFactory.SUPPORTED_ALGORITHMS["classification"]`` (or an
``include``/``exclude`` subset), evaluates each once via the model's own ``evaluate()``
on a single held-out partition (``val``, falling back to ``test`` — never ``oot``), and
ranks into a ``pd.DataFrame``. Per-algorithm failures are caught and recorded as a
``status="failed"`` row rather than aborting the whole comparison.

See :doc:`../pipeline` for how ``PipelineConfig.leaderboard.enabled=True`` wires this
into the full pipeline (the winner replaces ``model.algorithm`` for every later stage),
and :doc:`../packages/docs` for how the ranked table surfaces in the model card.
