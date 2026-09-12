dscompanion.models
==============

:meth:`dscompanion.ModelFactory.build` (documented in the :doc:`top-level API reference
<../api/dscompanion>`) is the single entry point for constructing a fitted-model wrapper —
it dispatches to the three task wrappers below (all subclassing ``BaseDSCompanionModel``)
and merges ``params`` on top of dscompanion's built-in per-algorithm defaults.
``ModelFactory.SUPPORTED_ALGORITHMS`` is the single source of truth for which algorithm
names are valid per task — it is also what ``PipelineConfig.model.algorithm`` validates
against and what :class:`dscompanion.Leaderboard` compares.

Currently supported (``ModelFactory.SUPPORTED_ALGORITHMS``):

- **classification**: xgboost, lightgbm, logistic, random_forest, gradient_boosting,
  svm, knn, decision_tree, extra_trees, adaboost, naive_bayes
- **regression**: xgboost, lightgbm, linear, ridge, lasso, elastic_net, random_forest,
  gradient_boosting, svm, knn, decision_tree, extra_trees, adaboost
- **clustering**: kmeans, dbscan, hierarchical

None of the classes below are re-exported at the top level — use e.g.
``dscompanion.models.ClassificationModel``.

.. currentmodule:: dscompanion.models

.. autosummary::
   :toctree: ../api/generated
   :nosignatures:

   BaseDSCompanionModel
   ClassificationModel
   RegressionModel
   ClusteringModel
