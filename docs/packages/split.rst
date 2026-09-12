dscompanion.split
============

Builds the train/val/test/OOT partitions every other stage consumes.
:class:`dscompanion.DataSplitter` and the :class:`dscompanion.DataSplit` it returns are documented
in the :doc:`top-level API reference <../api/dscompanion>` (both are re-exported at
``dscompanion.<Name>``). ``DataSplit`` is a plain dataclass (not a DataFrame subclass)
exposing both snake_case (``train_X``/``train_y``) and sklearn-style
(``X_train``/``y_train``) aliases as properties — ``X_test``/``y_test`` are always
present and non-empty; ``X_val``/``X_oot`` may be empty depending on ``split.method``.

``DataSplitMetadata`` (below) is split-internal and not re-exported at the top level.

.. currentmodule:: dscompanion.split

.. autosummary::
   :toctree: ../api/generated
   :nosignatures:

   DataSplitMetadata
