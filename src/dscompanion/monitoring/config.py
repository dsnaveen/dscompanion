"""MonitoringConfig: pydantic schema for a YAML-driven monitoring job.

Joins a past ScoringRunner output (scored_data) with a separately-arrived actuals
file, re-measures performance against the actual outcome, and — when raw_data is
supplied — computes feature-level drift (CSI) against the ScoringPipeline's
training-time reference. Minimum required fields: name, scoring_pipeline_path,
scored_data.path, actuals_data.path, actuals_data.target_column, id_columns.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from dscompanion.scoring.config import ScoringDataConfig

__all__ = ["MonitoringConfig", "MonitoringActualsConfig", "MonitoringOutputConfig"]


class MonitoringActualsConfig(ScoringDataConfig):
    """Specifies where to read the actuals file and which column holds the true outcome.

    Extends ``ScoringDataConfig`` (``path``/``format``/``sheet_name``,
    ``extra="forbid"`` inherited) with ``target_column``.

    Args:
        target_column (str): Name of the actual outcome column in this file.
            Required.
    """

    target_column: str


class MonitoringOutputConfig(BaseModel):
    """Specifies where and how the monitoring reports are written.

    Args:
        output_dir (str): Root directory this run's output lives under — a
            timestamped ``<output_dir>/<run_id>/`` folder is created per
            run, mirroring ``PipelineRunner``/``ScoringRunner``'s run-folder
            convention.
        format (str): File format for the written report tables. One of
            ``"csv"``, ``"parquet"``, ``"excel"``. Defaults to ``"csv"``.

    Returns:
        MonitoringOutputConfig: Validated output specification.
    """

    model_config = ConfigDict(extra="forbid")

    output_dir: str = "./monitoring_runs"
    format: str = "csv"

    @field_validator("format")
    @classmethod
    def valid_format(cls, v: str) -> str:
        allowed = {"csv", "parquet", "excel"}
        if v not in allowed:
            raise ValueError(f"format must be one of {allowed}, got {v!r}")
        return v


class MonitoringConfig(BaseModel):
    """Root configuration object for a YAML-driven monitoring job.

    Loaded from a YAML file by the scientist and passed directly to
    ``MonitoringRunner``. Validates all nested configs at construction time,
    rejecting typos and invalid values before a monitoring run runs.

    IMPORTANT constraint: ``id_columns`` here must match the ``id_columns``
    the ORIGINAL ``ScoringRunner`` run used to produce ``scored_data`` — if
    that run didn't set ``id_columns``, there is no join key between
    ``scored_data`` and ``actuals_data``/``raw_data``, and this config
    cannot be used meaningfully.

    Minimum viable YAML (everything else defaults)::

        name: my_model_v1_monitoring
        scoring_pipeline_path: ./reports/20260101_000000/model/v1.0_scoring_pipeline.joblib
        scored_data:
          path: ./scoring_runs/20260901_120000/scored.parquet
        actuals_data:
          path: abfss://container@account.dfs.core.windows.net/actuals/actuals.parquet
          target_column: actual_outcome
        id_columns:
          - row_id

    Args:
        name (str): Monitoring job name. Used as the run name. Required.
        version (str): Monitoring job version string. Defaults to ``"1.0"``.
        scoring_pipeline_path (str): Path to the same ``ScoringPipeline``
            bundle used to produce ``scored_data`` — supplies
            ``psi_reference_`` (for the ``"psi"`` performance metric) and
            ``feature_reference_`` (for CSI). Required.
        scored_data (ScoringDataConfig): A past ``ScoringRunner`` output
            file — must contain ``id_columns... , "prediction"`` and, for
            classification, ``"probability"`` (``ScoringPipeline.predict()``'s
            exact, fixed output column convention, not independently
            configurable). Required.
        actuals_data (MonitoringActualsConfig): The separately-arrived
            actual outcomes for that same batch — must contain
            ``id_columns...`` and ``target_column``. Required.
        raw_data (ScoringDataConfig | None): The same batch's raw feature
            data, for feature-level drift (CSI) via
            ``ScoringPipeline.compute_feature_drift()``. Optional — when
            omitted, only performance metrics are computed and no
            feature-drift report is produced. Defaults to ``None``.
        id_columns (list[str]): Join key(s) across ``scored_data``,
            ``actuals_data``, and ``raw_data``. Required, must be
            non-empty.
        output (MonitoringOutputConfig): Where/how reports are written.

    Returns:
        MonitoringConfig: Fully validated monitoring job configuration
        ready to pass to ``MonitoringRunner``.

    Raises:
        pydantic.ValidationError: If any required field is missing or any
            value fails validation.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str = "1.0"
    scoring_pipeline_path: str
    scored_data: ScoringDataConfig
    actuals_data: MonitoringActualsConfig
    raw_data: ScoringDataConfig | None = None
    id_columns: list[str]
    output: MonitoringOutputConfig = Field(default_factory=MonitoringOutputConfig)

    @field_validator("id_columns")
    @classmethod
    def non_empty_id_columns(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError(
                "id_columns must be a non-empty list — required to join scored_data, "
                "actuals_data, and (if set) raw_data by row identity. This means "
                "id_columns must also have been set on the ORIGINAL ScoringRunner run "
                "that produced scored_data, or there is no join key."
            )
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MonitoringConfig":
        """Load and validate a MonitoringConfig from a YAML file.

        Args:
            path (str | Path): Path to the monitoring YAML config file.

        Returns:
            MonitoringConfig: Fully validated monitoring configuration.

        Raises:
            FileNotFoundError: If the file does not exist at ``path``.
            yaml.YAMLError: If the file is not valid YAML.
            pydantic.ValidationError: If the config fails validation.
        """
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh)
        return cls(**raw)
