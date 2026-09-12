"""dscompanion — production-grade ML toolkit for financial modelling."""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from dscompanion.calibration import Calibrator
from dscompanion.config import settings
from dscompanion.docs import ModelCard
from dscompanion.eda import EDAReport
from dscompanion.explain import BootstrapSHAPExplainer, LIMEExplainer, SHAPExplainer
from dscompanion.features import FeatureProcessingPipeline
from dscompanion.leaderboard import Leaderboard
from dscompanion.models import ModelFactory
from dscompanion.pipeline import PipelineConfig, PipelineRunner, PipelineRunResult
from dscompanion.selection import FeatureSelectionPipeline
from dscompanion.split import DataSplit, DataSplitter
from dscompanion.targets import ImbalanceHandler, TargetBinariser
from dscompanion.tracking import tracking_run
from dscompanion.tuning import Tuner
from dscompanion.utils.synthetic import SyntheticDataGenerator

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Split
    "DataSplitter",
    "DataSplit",
    # EDA
    "EDAReport",
    # Features
    "FeatureProcessingPipeline",
    # Targets
    "ImbalanceHandler",
    "TargetBinariser",
    # Selection
    "FeatureSelectionPipeline",
    # Models
    "ModelFactory",
    # Leaderboard
    "Leaderboard",
    # Tuning
    "Tuner",
    # Explainability
    "SHAPExplainer",
    "BootstrapSHAPExplainer",
    "LIMEExplainer",
    # Calibration
    "Calibrator",
    # Docs
    "ModelCard",
    # Tracking
    "tracking_run",
    # Config
    "settings",
    # Pipeline
    "PipelineConfig",
    "PipelineRunner",
    "PipelineRunResult",
    # Utils
    "SyntheticDataGenerator",
]
