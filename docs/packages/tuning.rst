dscompanion.tuning
==============

:class:`dscompanion.Tuner` is documented in the :doc:`top-level API reference
<../api/dscompanion>` (it's re-exported at the top level).

``Tuner(model, backend="optuna", n_trials=..., cv=<DataSplit>, metric=..., direction=...)``
runs a hyperparameter search using the search space registered for that model's
algorithm in ``SEARCH_SPACES`` (keyed ``"{backend}"`` → ``"{algorithm}_{task}"``), and
returns a **new** fitted model with the best parameters — not the ``Tuner`` itself.
``Tuner._infer_space_key()`` resolves the search-space key from the fitted estimator's
class name, requiring every token of a candidate key to match (not just one) to avoid
collisions like ``"tree"`` falsely matching ``ExtraTreesClassifier``.
``PREDEFINED_PARAMS`` supplies fixed (non-searched) parameter sets for the
``"predefined"`` backend.

``SEARCH_SPACES`` and ``PREDEFINED_PARAMS`` are module-level dicts (one entry per
``{algorithm}_{task}`` key, per backend), not re-exported at the top level — see
``dscompanion/tuning/search_spaces.py`` directly for the full parameter grids.
