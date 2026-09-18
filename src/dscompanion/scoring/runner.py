"""ScoringRunner: YAML-driven batch scoring executor, the scoring-side counterpart
to PipelineRunner.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from dscompanion.pipeline.loaders import load_raw_data
from dscompanion.pipeline.run_utils import (
    attach_run_log_handler,
    detach_run_log_handler,
    generate_run_id_and_dir,
)
from dscompanion.scoring.config import ScoringConfig
from dscompanion.scoring.pipeline import ScoringPipeline

logger = logging.getLogger(__name__)

__all__ = ["ScoringRunner", "ScoringRunResult"]


@dataclass
class ScoringRunResult:
    """All artefacts produced by a completed scoring run.

    Returned by ``ScoringRunner.run()``.

    Args:
        config (ScoringConfig): The full validated config that drove this run.
        scored_df (pd.DataFrame): The scored output — same shape
            ``ScoringPipeline.predict()`` returns (``id_columns... ,
            "prediction"`` and, for classification, ``"probability"``).
        drift_report (pd.DataFrame | None): ``ScoringPipeline.compute_drift()``'s
            output, or ``None`` when ``config.check_drift=False``.
        run_id (str): This run's id (``<name>_yyyymmdd_hhmmss`` — see
            ``dscompanion.pipeline.run_utils.generate_run_id_and_dir``).
        run_dir (Path): ``<output.output_dir>/<run_id>/`` — this run's
            output directory.
        output_path (Path): Path to the written scored output file, inside
            ``run_dir``.
        log_path (Path): Path to this run's captured log file, inside
            ``run_dir``.
        elapsed_seconds (float): Wall-clock seconds from ``run()`` start to
            finish.

    Returns:
        ScoringRunResult: Fully populated result container.
    """

    config: ScoringConfig
    scored_df: pd.DataFrame
    drift_report: pd.DataFrame | None
    run_id: str
    run_dir: Path
    output_path: Path
    log_path: Path
    elapsed_seconds: float = 0.0


class ScoringRunner:
    """Execute a YAML-driven batch scoring job and return all artefacts.

    Loads a ``ScoringPipeline`` bundle produced by a prior ``PipelineRunner``
    run, scores new raw data against it, and writes the result into a
    timestamped ``<output.output_dir>/<run_id>/`` folder — the same
    name-prefixed, timestamped, audited run-folder convention
    ``PipelineRunner`` uses for training, so a scoring job is just as
    traceable/reproducible.

    Args:
        config (ScoringConfig): Validated scoring configuration.

    Example::

        result = ScoringRunner.from_yaml("scoring/my_model_v1_scoring.yaml").run()
        print(result.scored_df)
        print(result.output_path)
    """

    def __init__(self, config: ScoringConfig) -> None:
        self.config = config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ScoringRunner":
        """Construct a ``ScoringRunner`` by loading and validating a YAML config file.

        Args:
            path (str | Path): Path to the scoring YAML config file.

        Returns:
            ScoringRunner: Ready to call ``.run()`` on.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            pydantic.ValidationError: If the config is invalid.
        """
        return cls(ScoringConfig.from_yaml(path))

    def run(self) -> ScoringRunResult:
        """Load the scoring bundle + new data, score, and write the result.

        Args:
            None

        Returns:
            ScoringRunResult: The scored DataFrame, optional drift report,
            and this run's output paths.

        Raises:
            RuntimeError: If loading the input data fails, or the
                ``ScoringPipeline`` bundle's ``bundle_schema_version``
                doesn't match this ``dscompanion`` version.
            ValueError: If the new data fails ``ScoringPipeline.predict()``'s
                schema validation (missing/incompatible columns, empty
                input) — see ``dscompanion.scoring.ScoringPipeline.predict``.
        """
        t0 = time.time()
        cfg = self.config
        logger.info("=" * 60)
        logger.info("ScoringRunner  |  %s  v%s", cfg.name, cfg.version)
        logger.info("=" * 60)

        run_id, run_dir = generate_run_id_and_dir(cfg.output.output_dir, name=cfg.name)
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Run directory: %s", run_dir)

        log_path = run_dir / "scoring.log"
        handler, prev_level = attach_run_log_handler(log_path)
        try:
            logger.info("[1/3] Loading data from %s", cfg.data.path)
            df = load_raw_data(cfg.data.path, cfg.data.format, cfg.data.sheet_name)
            logger.info("      Loaded %d rows × %d columns", len(df), len(df.columns))

            logger.info("[2/3] Loading ScoringPipeline from %s", cfg.scoring_pipeline_path)
            scoring_pipeline = ScoringPipeline.load(cfg.scoring_pipeline_path)

            logger.info("[3/3] Scoring")
            scored_df = scoring_pipeline.predict(df, id_columns=cfg.id_columns)

            drift_report = None
            if cfg.check_drift:
                logger.info("      Computing drift report")
                drift_report = scoring_pipeline.compute_drift(df)
                drift_path = run_dir / "drift_report.csv"
                drift_report.to_csv(drift_path, index=False)
                logger.info("      Drift report: %s", drift_path)

            output_path = self._write_output(scored_df, run_dir, cfg.output.format)
            logger.info("      Scored output: %s", output_path)

            config_path = run_dir / "config.yaml"
            with open(config_path, "w") as fh:
                yaml.dump(cfg.model_dump(), fh, default_flow_style=False, sort_keys=False)

            elapsed = time.time() - t0
            logger.info("=" * 60)
            logger.info("Scoring complete — %.1fs  |  Run: %s", elapsed, run_id)
            logger.info("=" * 60)

            return ScoringRunResult(
                config=cfg,
                scored_df=scored_df,
                drift_report=drift_report,
                run_id=run_id,
                run_dir=run_dir,
                output_path=output_path,
                log_path=log_path,
                elapsed_seconds=elapsed,
            )
        finally:
            detach_run_log_handler(handler, prev_level)

    def _write_output(self, scored_df: pd.DataFrame, run_dir: Path, fmt: str) -> Path:
        """Write the scored output to ``<run_dir>/scored.<ext>``.

        Resets the index into an explicit column first (rather than
        ``index=False`` discarding it outright) — the index is how
        ``ScoringPipeline.predict()`` keeps output aligned to input, and
        silently dropping it would lose that alignment for callers who
        didn't pass ``id_columns``.

        Args:
            scored_df (pd.DataFrame): Output from ``ScoringPipeline.predict()``.
            run_dir (Path): This run's output directory.
            fmt (str): One of ``"parquet"``, ``"csv"``, ``"excel"``.

        Returns:
            Path: The path written to.
        """
        ext = "xlsx" if fmt == "excel" else fmt
        path = run_dir / f"scored.{ext}"
        out = scored_df.reset_index()
        if fmt == "parquet":
            out.to_parquet(path, index=False)
        elif fmt == "csv":
            out.to_csv(path, index=False)
        else:  # excel
            out.to_excel(path, index=False)
        return path
