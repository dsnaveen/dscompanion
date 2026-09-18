"""ScoringConfig: pydantic schema for a YAML-driven batch scoring job.

A scientist writes a YAML, passes it to ScoringRunner, and gets a scored
output file. Minimum required fields: name, scoring_pipeline_path, data.path,
output.output_dir.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

__all__ = ["ScoringConfig", "ScoringDataConfig", "ScoringOutputConfig"]


class ScoringDataConfig(BaseModel):
    """Specifies where to read the new, unseen data to be scored.

    Mirrors the loading-relevant subset of ``dscompanion.pipeline.config.DataConfig``
    — no ``target``/``feature_columns``/``ignore_columns``/``date_column``, since
    scoring input has no label and no column-selection step (``ScoringPipeline``
    already knows, from training, exactly which raw columns it needs).

    Args:
        path (str): Full path to the input data file or Delta table.
        format (str): File format. One of ``"parquet"``, ``"csv"``,
            ``"excel"``, ``"delta"``. Defaults to ``"parquet"``.
        sheet_name (str | int | list[str | int] | None): Only used when
            ``format="excel"``; ignored otherwise. See ``DataConfig.sheet_name``.

    Returns:
        ScoringDataConfig: Validated data specification.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    format: str = "parquet"
    sheet_name: str | int | list[str | int] | None = None

    @field_validator("format")
    @classmethod
    def valid_format(cls, v: str) -> str:
        allowed = {"parquet", "csv", "excel", "delta"}
        if v not in allowed:
            raise ValueError(f"format must be one of {allowed}, got {v!r}")
        return v


class ScoringOutputConfig(BaseModel):
    """Specifies where and how the scored output is written.

    Args:
        output_dir (str): Root directory this scoring run's output lives
            under — a timestamped ``<output_dir>/<run_id>/`` folder is
            created per run, mirroring ``PipelineRunner``'s run-folder
            convention.
        format (str): File format for the scored output. One of
            ``"parquet"``, ``"csv"``, ``"excel"``. Defaults to ``"parquet"``.

    Returns:
        ScoringOutputConfig: Validated output specification.
    """

    model_config = ConfigDict(extra="forbid")

    output_dir: str = "./scoring_runs"
    format: str = "parquet"

    @field_validator("format")
    @classmethod
    def valid_format(cls, v: str) -> str:
        allowed = {"parquet", "csv", "excel"}
        if v not in allowed:
            raise ValueError(f"format must be one of {allowed}, got {v!r}")
        return v


class ScoringConfig(BaseModel):
    """Root configuration object for a YAML-driven batch scoring job.

    Loaded from a YAML file by the scientist and passed directly to
    ``ScoringRunner``. Validates all nested configs at construction time,
    rejecting typos and invalid values before scoring runs.

    Minimum viable YAML (everything else defaults)::

        name: my_model_v1_scoring
        scoring_pipeline_path: ./reports/20260101_000000/model/v1.0_scoring_pipeline.joblib
        data:
          path: abfss://container@account.dfs.core.windows.net/features/new_applications.parquet
        output:
          output_dir: ./scoring_runs

    Args:
        name (str): Scoring job name. Used as the run name. Required.
        version (str): Scoring job version string. Defaults to ``"1.0"``.
        scoring_pipeline_path (str): Path to a ``ScoringPipeline`` bundle
            saved by a prior ``PipelineRunner`` run
            (``PipelineRunResult.scoring_pipeline_path``). Required.
        data (ScoringDataConfig): New, unseen data to score. Required.
        id_columns (list[str] | None): Raw columns to carry through to the
            scored output unchanged (e.g. a customer/account identifier).
            Passed straight to ``ScoringPipeline.predict()``. Defaults to
            ``None``.
        check_drift (bool): When ``True``, also runs
            ``ScoringPipeline.compute_drift()`` on this batch and saves the
            report alongside the scored output. Defaults to ``False``.
        output (ScoringOutputConfig): Where/how the scored output is
            written.

    Returns:
        ScoringConfig: Fully validated scoring job configuration ready to
        pass to ``ScoringRunner``.

    Raises:
        pydantic.ValidationError: If any required field is missing or any
            value fails validation.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str = "1.0"
    scoring_pipeline_path: str
    data: ScoringDataConfig
    id_columns: list[str] | None = None
    check_drift: bool = False
    output: ScoringOutputConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ScoringConfig":
        """Load and validate a ScoringConfig from a YAML file.

        Args:
            path (str | Path): Path to the scoring YAML config file.

        Returns:
            ScoringConfig: Fully validated scoring configuration.

        Raises:
            FileNotFoundError: If the file does not exist at ``path``.
            yaml.YAMLError: If the file is not valid YAML.
            pydantic.ValidationError: If the config fails validation.
        """
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh)
        return cls(**raw)
