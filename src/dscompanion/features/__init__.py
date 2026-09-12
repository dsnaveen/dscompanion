"""dscompanion.features — feature engineering sub-package."""

from dscompanion.features.binner import AutoBinner
from dscompanion.features.combiner import CategoryCombinerTransformer, ColumnArithmeticTransformer
from dscompanion.features.compression import PCATransformer
from dscompanion.features.date_features import DateFeatureExtractor
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
from dscompanion.features.imputer import MultivariateImputer, SmartImputer
from dscompanion.features.leakage_guard import LeakageError, LeakageGuard, LeakageReport
from dscompanion.features.noise import NoiseInjector
from dscompanion.features.pipeline import FeatureProcessingPipeline
from dscompanion.features.polynomial import PolynomialFeaturesTransformer
from dscompanion.features.rank import PercentileRankTransformer
from dscompanion.features.recommend import recommend_column_recipe
from dscompanion.features.registry import (
    TRANSFORMER_REGISTRY,
    build_transformer,
    registered_transformer_names,
)
from dscompanion.features.relative import GroupRelativeTransformer
from dscompanion.features.scaler import (
    SmartScaler,
    StatisticalOutlierCapper,
    WinsorizationTransformer,
)
from dscompanion.features.spline import SplineFeatureTransformer
from dscompanion.features.transform_chain import ColumnRecipe, FeatureTransformChain, TransformStep

__all__ = [
    "TRANSFORMER_REGISTRY",
    "AutoBinner",
    "BinaryEncoder",
    "CategoryCombinerTransformer",
    "ColumnArithmeticTransformer",
    "ColumnRecipe",
    "DateFeatureExtractor",
    "DistributionTransformer",
    "FeatureProcessingPipeline",
    "FeatureTransformChain",
    "GroupRelativeTransformer",
    "HashEncoder",
    "HighCardinalityEncoder",
    "LeakageError",
    "LeakageGuard",
    "LeakageReport",
    "MultivariateImputer",
    "NoiseInjector",
    "OneHotEncoder",
    "OrdinalEncoder",
    "PCATransformer",
    "PercentileRankTransformer",
    "PolynomialFeaturesTransformer",
    "RareCategoryGrouper",
    "SmartImputer",
    "SmartScaler",
    "SplineFeatureTransformer",
    "StatisticalOutlierCapper",
    "TransformStep",
    "WinsorizationTransformer",
    "WoEEncoder",
    "build_transformer",
    "recommend_column_recipe",
    "registered_transformer_names",
]
