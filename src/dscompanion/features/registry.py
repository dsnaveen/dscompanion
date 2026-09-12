"""Transformer registry for per-feature transformation chains.

Maps a step name (as referenced by a ``TransformStep`` in a ``ColumnRecipe``,
see ``dscompanion.features.transform_chain``) to a factory that builds an
unfitted dscompanion transformer instance scoped to operate correctly on a
single-column ``pd.DataFrame`` slice.

Excludes ``GroupRelativeTransformer``, ``PCATransformer``, and
``MultivariateImputer`` — all three are inherently multi-column/cross-column
(a group-by key from a *different* column, compression of several columns
into few, or cross-column imputation), not single-column recipe steps.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

from sklearn.base import BaseEstimator, TransformerMixin

from dscompanion.features.binner import AutoBinner
from dscompanion.features.distribution import DistributionTransformer
from dscompanion.features.encoder import (
    BinaryEncoder,
    HashEncoder,
    HighCardinalityEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    RareCategoryGrouper,
    WoEEncoder,
)
from dscompanion.features.imputer import SmartImputer
from dscompanion.features.noise import NoiseInjector
from dscompanion.features.polynomial import PolynomialFeaturesTransformer
from dscompanion.features.rank import PercentileRankTransformer
from dscompanion.features.scaler import (
    SmartScaler,
    StatisticalOutlierCapper,
    WinsorizationTransformer,
)
from dscompanion.features.spline import SplineFeatureTransformer

__all__ = ["TRANSFORMER_REGISTRY", "build_transformer", "registered_transformer_names"]


def _impute(**params: Any) -> SmartImputer:
    kwargs: dict[str, Any] = {"add_missing_indicator": False}
    kwargs.update(params)
    return SmartImputer(**kwargs)


def _clip_lower(**params: Any) -> WinsorizationTransformer:
    kwargs: dict[str, Any] = {"upper": 0.0}
    kwargs.update(params)
    return WinsorizationTransformer(**kwargs)


def _clip_upper(**params: Any) -> WinsorizationTransformer:
    kwargs: dict[str, Any] = {"lower": 0.0}
    kwargs.update(params)
    return WinsorizationTransformer(**kwargs)


def _clip_both(**params: Any) -> WinsorizationTransformer:
    return WinsorizationTransformer(**params)


def _clip_iqr(**params: Any) -> StatisticalOutlierCapper:
    kwargs: dict[str, Any] = {"method": "iqr"}
    kwargs.update(params)
    return StatisticalOutlierCapper(**kwargs)


def _clip_zscore(**params: Any) -> StatisticalOutlierCapper:
    kwargs: dict[str, Any] = {"method": "zscore"}
    kwargs.update(params)
    return StatisticalOutlierCapper(**kwargs)


def _clip_mad(**params: Any) -> StatisticalOutlierCapper:
    kwargs: dict[str, Any] = {"method": "mad"}
    kwargs.update(params)
    return StatisticalOutlierCapper(**kwargs)


def _log(**params: Any) -> DistributionTransformer:
    return DistributionTransformer(strategy="log", **params)


def _log1p(**params: Any) -> DistributionTransformer:
    return DistributionTransformer(strategy="log1p", **params)


def _sqrt(**params: Any) -> DistributionTransformer:
    return DistributionTransformer(strategy="sqrt", **params)


def _cbrt(**params: Any) -> DistributionTransformer:
    return DistributionTransformer(strategy="cbrt", **params)


def _reciprocal(**params: Any) -> DistributionTransformer:
    return DistributionTransformer(strategy="reciprocal", **params)


def _yeo_johnson(**params: Any) -> DistributionTransformer:
    return DistributionTransformer(strategy="yeo_johnson", **params)


def _quantile_uniform(**params: Any) -> DistributionTransformer:
    return DistributionTransformer(strategy="quantile_uniform", **params)


def _quantile_normal(**params: Any) -> DistributionTransformer:
    return DistributionTransformer(strategy="quantile_normal", **params)


def _bucket_quantile(**params: Any) -> AutoBinner:
    return AutoBinner(strategy="quantile", **params)


def _bucket_uniform(**params: Any) -> AutoBinner:
    return AutoBinner(strategy="uniform", **params)


def _bucket_tree(**params: Any) -> AutoBinner:
    return AutoBinner(strategy="tree", **params)


def _percentile_rank(**params: Any) -> PercentileRankTransformer:
    return PercentileRankTransformer(**params)


def _dense_rank(**params: Any) -> PercentileRankTransformer:
    kwargs: dict[str, Any] = {"method": "dense"}
    kwargs.update(params)
    return PercentileRankTransformer(**kwargs)


def _global_rank(**params: Any) -> PercentileRankTransformer:
    kwargs: dict[str, Any] = {"method": "global"}
    kwargs.update(params)
    return PercentileRankTransformer(**kwargs)


def _scale(**params: Any) -> SmartScaler:
    return SmartScaler(**params)


def _ordinal_encode(**params: Any) -> OrdinalEncoder:
    return OrdinalEncoder(**params)


def _onehot_encode(**params: Any) -> OneHotEncoder:
    return OneHotEncoder(**params)


def _rare_group(**params: Any) -> RareCategoryGrouper:
    return RareCategoryGrouper(**params)


def _target_encode(**params: Any) -> HighCardinalityEncoder:
    kwargs: dict[str, Any] = {"strategy": "target_encoding"}
    kwargs.update(params)
    return HighCardinalityEncoder(**kwargs)


def _frequency_encode(**params: Any) -> HighCardinalityEncoder:
    kwargs: dict[str, Any] = {"strategy": "frequency_encoding"}
    kwargs.update(params)
    return HighCardinalityEncoder(**kwargs)


def _woe_encode(**params: Any) -> WoEEncoder:
    return WoEEncoder(**params)


def _binary_encode(**params: Any) -> BinaryEncoder:
    return BinaryEncoder(**params)


def _hash_encode(**params: Any) -> HashEncoder:
    return HashEncoder(**params)


def _polynomial(**params: Any) -> PolynomialFeaturesTransformer:
    return PolynomialFeaturesTransformer(**params)


def _spline(**params: Any) -> SplineFeatureTransformer:
    return SplineFeatureTransformer(**params)


def _noise(**params: Any) -> NoiseInjector:
    return NoiseInjector(**params)


TRANSFORMER_REGISTRY: dict[str, Callable[..., BaseEstimator | TransformerMixin]] = {
    "impute": _impute,
    "clip_lower": _clip_lower,
    "clip_upper": _clip_upper,
    "clip_both": _clip_both,
    "clip_iqr": _clip_iqr,
    "clip_zscore": _clip_zscore,
    "clip_mad": _clip_mad,
    "log": _log,
    "log1p": _log1p,
    "sqrt": _sqrt,
    "cbrt": _cbrt,
    "reciprocal": _reciprocal,
    "yeo_johnson": _yeo_johnson,
    "quantile_uniform": _quantile_uniform,
    "quantile_normal": _quantile_normal,
    "bucket_quantile": _bucket_quantile,
    "bucket_uniform": _bucket_uniform,
    "bucket_tree": _bucket_tree,
    "percentile_rank": _percentile_rank,
    "dense_rank": _dense_rank,
    "global_rank": _global_rank,
    "scale": _scale,
    "ordinal_encode": _ordinal_encode,
    "onehot_encode": _onehot_encode,
    "rare_group": _rare_group,
    "target_encode": _target_encode,
    "frequency_encode": _frequency_encode,
    "woe_encode": _woe_encode,
    "binary_encode": _binary_encode,
    "hash_encode": _hash_encode,
    "polynomial": _polynomial,
    "spline": _spline,
    "noise": _noise,
}


def build_transformer(name: str, params: dict[str, Any] | None = None) -> BaseEstimator:
    """Build an unfitted dscompanion transformer instance from a registered step name.

    Args:
        name (str): Registry key — see ``registered_transformer_names()`` for
            the full list of valid values.
        params (dict[str, Any] | None): Keyword arguments forwarded to the
            transformer's constructor, overriding its defaults. Defaults to
            ``None`` (use the transformer's own default parameters).

    Returns:
        BaseEstimator: A fresh, unfitted transformer instance implementing
        the sklearn ``fit``/``transform`` interface.

    Raises:
        ValueError: If ``name`` is not a registered transformer name.
    """
    if name not in TRANSFORMER_REGISTRY:
        raise ValueError(
            f"Unknown transform step {name!r} — must be one of " f"{registered_transformer_names()}"
        )
    return TRANSFORMER_REGISTRY[name](**(params or {}))


def registered_transformer_names() -> list[str]:
    """Return every registered transform-step name, sorted alphabetically.

    Args:
        None

    Returns:
        list[str]: All valid ``TransformStep.transformer`` values.
    """
    return sorted(TRANSFORMER_REGISTRY)
