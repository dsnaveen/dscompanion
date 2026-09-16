"""Tests for dscompanion.scoring.ScoringConfig."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from dscompanion.scoring.config import ScoringConfig, ScoringDataConfig, ScoringOutputConfig


def _minimal_kwargs(**overrides: object) -> dict:
    base = {
        "name": "test_scoring",
        "scoring_pipeline_path": "model.joblib",
        "data": ScoringDataConfig(path="new_data.parquet"),
        "output": ScoringOutputConfig(output_dir="./scoring_runs"),
    }
    base.update(overrides)
    return base


class TestScoringConfigValidation:
    def test_minimal_config_constructs_with_defaults(self):
        cfg = ScoringConfig(**_minimal_kwargs())
        assert cfg.version == "1.0"
        assert cfg.id_columns is None
        assert cfg.check_drift is False
        assert cfg.data.format == "parquet"
        assert cfg.output.format == "parquet"

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            ScoringConfig(name="x", data=ScoringDataConfig(path="y.parquet"))

    def test_unknown_top_level_field_rejected(self):
        with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
            ScoringConfig(**_minimal_kwargs(bogus_field=True))

    def test_id_columns_and_check_drift_pass_through(self):
        cfg = ScoringConfig(**_minimal_kwargs(id_columns=["customer_id"], check_drift=True))
        assert cfg.id_columns == ["customer_id"]
        assert cfg.check_drift is True


class TestScoringDataConfigValidation:
    def test_invalid_format_raises(self):
        with pytest.raises(ValidationError, match="format must be one of"):
            ScoringDataConfig(path="x", format="xml")

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
            ScoringDataConfig(path="x", bogus=True)


class TestScoringOutputConfigValidation:
    def test_invalid_format_raises(self):
        with pytest.raises(ValidationError, match="format must be one of"):
            ScoringOutputConfig(output_dir="./x", format="delta")

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
            ScoringOutputConfig(output_dir="./x", bogus=True)


class TestFromYaml:
    def test_from_yaml_round_trip(self, tmp_path):
        yaml_path = tmp_path / "scoring.yaml"
        yaml_path.write_text(
            yaml.dump(
                {
                    "name": "yaml_test",
                    "scoring_pipeline_path": "model.joblib",
                    "data": {"path": "new_data.parquet", "format": "parquet"},
                    "id_columns": ["customer_id"],
                    "check_drift": True,
                    "output": {"output_dir": "./scoring_runs", "format": "csv"},
                }
            )
        )
        cfg = ScoringConfig.from_yaml(yaml_path)
        assert cfg.name == "yaml_test"
        assert cfg.id_columns == ["customer_id"]
        assert cfg.check_drift is True
        assert cfg.output.format == "csv"

    def test_from_yaml_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ScoringConfig.from_yaml(tmp_path / "does_not_exist.yaml")
