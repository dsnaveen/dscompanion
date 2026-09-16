"""Tests for dscompanion.monitoring.MonitoringConfig."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from dscompanion.monitoring.config import (
    MonitoringActualsConfig,
    MonitoringConfig,
    MonitoringOutputConfig,
)
from dscompanion.scoring.config import ScoringDataConfig


def _minimal_kwargs(**overrides: object) -> dict:
    base = {
        "name": "test_monitoring",
        "scoring_pipeline_path": "model.joblib",
        "scored_data": ScoringDataConfig(path="scored.parquet"),
        "actuals_data": MonitoringActualsConfig(path="actuals.parquet", target_column="target"),
        "id_columns": ["customer_id"],
        "output": MonitoringOutputConfig(output_dir="./monitoring_runs"),
    }
    base.update(overrides)
    return base


class TestMonitoringConfigValidation:
    def test_minimal_config_constructs_with_defaults(self):
        cfg = MonitoringConfig(**_minimal_kwargs())
        assert cfg.version == "1.0"
        assert cfg.raw_data is None
        assert cfg.output.format == "csv"

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            MonitoringConfig(name="x", scored_data=ScoringDataConfig(path="y.parquet"))

    def test_unknown_top_level_field_rejected(self):
        with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
            MonitoringConfig(**_minimal_kwargs(bogus_field=True))

    def test_empty_id_columns_raises(self):
        with pytest.raises(ValidationError, match="non-empty"):
            MonitoringConfig(**_minimal_kwargs(id_columns=[]))

    def test_raw_data_optional_and_can_be_set(self):
        cfg = MonitoringConfig(**_minimal_kwargs(raw_data=ScoringDataConfig(path="raw.parquet")))
        assert cfg.raw_data is not None
        assert cfg.raw_data.path == "raw.parquet"


class TestMonitoringActualsConfigValidation:
    def test_requires_target_column(self):
        with pytest.raises(ValidationError):
            MonitoringActualsConfig(path="x.parquet")

    def test_inherits_format_validation(self):
        with pytest.raises(ValidationError, match="format must be one of"):
            MonitoringActualsConfig(path="x", format="xml", target_column="target")

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
            MonitoringActualsConfig(path="x", target_column="target", bogus=True)


class TestMonitoringOutputConfigValidation:
    def test_invalid_format_raises(self):
        with pytest.raises(ValidationError, match="format must be one of"):
            MonitoringOutputConfig(output_dir="./x", format="delta")

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
            MonitoringOutputConfig(output_dir="./x", bogus=True)


class TestFromYaml:
    def test_from_yaml_round_trip_without_raw_data(self, tmp_path):
        yaml_path = tmp_path / "monitoring.yaml"
        yaml_path.write_text(
            yaml.dump(
                {
                    "name": "yaml_test",
                    "scoring_pipeline_path": "model.joblib",
                    "scored_data": {"path": "scored.parquet", "format": "parquet"},
                    "actuals_data": {
                        "path": "actuals.parquet",
                        "format": "parquet",
                        "target_column": "default_flag",
                    },
                    "id_columns": ["customer_id"],
                    "output": {"output_dir": "./monitoring_runs", "format": "csv"},
                }
            )
        )
        cfg = MonitoringConfig.from_yaml(yaml_path)
        assert cfg.name == "yaml_test"
        assert cfg.raw_data is None
        assert cfg.actuals_data.target_column == "default_flag"

    def test_from_yaml_round_trip_with_raw_data(self, tmp_path):
        yaml_path = tmp_path / "monitoring.yaml"
        yaml_path.write_text(
            yaml.dump(
                {
                    "name": "yaml_test",
                    "scoring_pipeline_path": "model.joblib",
                    "scored_data": {"path": "scored.parquet"},
                    "actuals_data": {"path": "actuals.parquet", "target_column": "default_flag"},
                    "raw_data": {"path": "raw.parquet"},
                    "id_columns": ["customer_id"],
                    "output": {"output_dir": "./monitoring_runs"},
                }
            )
        )
        cfg = MonitoringConfig.from_yaml(yaml_path)
        assert cfg.raw_data is not None
        assert cfg.raw_data.path == "raw.parquet"

    def test_from_yaml_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MonitoringConfig.from_yaml(tmp_path / "does_not_exist.yaml")
