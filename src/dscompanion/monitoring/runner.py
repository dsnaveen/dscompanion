"""MonitoringRunner: YAML-driven monitoring executor — re-measures performance against
actuals and computes feature-level drift (CSI) for a past scored batch, the
monitoring-side counterpart to ScoringRunner.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from dscompanion.config import settings
from dscompanion.models.classification import ClassificationModel
from dscompanion.models.regression import RegressionModel
from dscompanion.monitoring.config import MonitoringConfig
from dscompanion.pipeline.loaders import load_raw_data
from dscompanion.pipeline.run_utils import (
    attach_run_log_handler,
    detach_run_log_handler,
    generate_run_id_and_dir,
)
from dscompanion.scoring.pipeline import ScoringPipeline

logger = logging.getLogger(__name__)

__all__ = ["MonitoringRunner", "MonitoringRunResult"]


@dataclass
class MonitoringRunResult:
    """All artefacts produced by a completed monitoring run.

    Returned by ``MonitoringRunner.run()``.

    Args:
        config (MonitoringConfig): The full validated config that drove this
            run.
        performance_report (pd.DataFrame): Columns ``"split"`` (always
            ``"monitoring"``), ``"metric"``, ``"value"`` — same tidy shape
            as ``BaseDSCompanionModel.evaluate()``'s output, computed via
            the exact same ``_compute_metrics`` staticmethod training used.
        feature_drift_report (pd.DataFrame | None):
            ``ScoringPipeline.compute_feature_drift()``'s output on the
            matched ``raw_data`` rows, or ``None`` when ``config.raw_data``
            is unset or the ``raw_data`` join matched zero rows.
        n_scored_rows (int): Rows loaded from ``scored_data``.
        n_actuals_rows (int): Rows loaded from ``actuals_data``.
        n_matched_rows (int): Rows surviving the ``id_columns`` inner join
            between ``scored_data`` and ``actuals_data`` — the population
            performance metrics are computed over.
        run_id (str): This run's id (``yyyymmdd_hhmmss``, IST).
        run_dir (Path): ``<output.output_dir>/<run_id>/`` — this run's
            output directory.
        performance_report_path (Path): Path to the written performance
            report file, inside ``run_dir``.
        feature_drift_report_path (Path | None): Path to the written
            feature-drift report file, inside ``run_dir``, or ``None`` when
            ``feature_drift_report`` is ``None``.
        log_path (Path): Path to this run's captured log file, inside
            ``run_dir``.
        elapsed_seconds (float): Wall-clock seconds from ``run()`` start to
            finish.

    Returns:
        MonitoringRunResult: Fully populated result container.
    """

    config: MonitoringConfig
    performance_report: pd.DataFrame
    feature_drift_report: pd.DataFrame | None
    n_scored_rows: int
    n_actuals_rows: int
    n_matched_rows: int
    run_id: str
    run_dir: Path
    performance_report_path: Path
    feature_drift_report_path: Path | None
    log_path: Path
    elapsed_seconds: float = 0.0


class MonitoringRunner:
    """Execute a YAML-driven monitoring job and return all artefacts.

    Loads a ``ScoringPipeline`` bundle produced by a prior ``PipelineRunner``
    run, joins a past scored batch with its now-available actual outcomes,
    re-measures performance, and (when ``raw_data`` is set) computes
    feature-level drift (CSI) — writing the result into a timestamped
    ``<output.output_dir>/<run_id>/`` folder, the same IST-timestamped,
    audited run-folder convention ``PipelineRunner``/``ScoringRunner`` use.

    Args:
        config (MonitoringConfig): Validated monitoring configuration.

    Example::

        result = MonitoringRunner.from_yaml("monitoring/credit_risk_v1_monitoring.yaml").run()
        print(result.performance_report)
        print(result.feature_drift_report)
    """

    def __init__(self, config: MonitoringConfig) -> None:
        self.config = config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MonitoringRunner":
        """Construct a ``MonitoringRunner`` by loading and validating a YAML config file.

        Args:
            path (str | Path): Path to the monitoring YAML config file.

        Returns:
            MonitoringRunner: Ready to call ``.run()`` on.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            pydantic.ValidationError: If the config is invalid.
        """
        return cls(MonitoringConfig.from_yaml(path))

    def run(self) -> MonitoringRunResult:
        """Load scored+actuals(+raw) data, join, re-measure performance, compute
        feature drift, and write the result.

        Args:
            None

        Returns:
            MonitoringRunResult: The re-measured performance report, optional
            feature-drift report, and this run's output paths.

        Raises:
            RuntimeError: If loading any input data fails, or the
                ``ScoringPipeline`` bundle's ``bundle_schema_version``
                doesn't match this ``dscompanion`` version.
            ValueError: If the ``scored_data``/``actuals_data`` join on
                ``id_columns`` matches zero rows (nothing to monitor), or if
                ``raw_data`` is set but the loaded ``ScoringPipeline`` has no
                ``feature_reference_`` (see
                ``ScoringPipeline.compute_feature_drift``).
        """
        t0 = time.time()
        cfg = self.config
        logger.info("=" * 60)
        logger.info("MonitoringRunner  |  %s  v%s", cfg.name, cfg.version)
        logger.info("=" * 60)

        run_id, run_dir = generate_run_id_and_dir(cfg.output.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Run directory: %s", run_dir)

        log_path = run_dir / "monitoring.log"
        handler, prev_level = attach_run_log_handler(log_path)
        try:
            logger.info("[1/6] Loading scored data from %s", cfg.scored_data.path)
            scored_df = load_raw_data(
                cfg.scored_data.path, cfg.scored_data.format, cfg.scored_data.sheet_name
            )

            logger.info("[2/6] Loading actuals data from %s", cfg.actuals_data.path)
            actuals_df = load_raw_data(
                cfg.actuals_data.path, cfg.actuals_data.format, cfg.actuals_data.sheet_name
            )

            logger.info("[3/6] Loading ScoringPipeline from %s", cfg.scoring_pipeline_path)
            scoring_pipeline = ScoringPipeline.load(cfg.scoring_pipeline_path)

            logger.info("[4/6] Joining scored + actuals on id_columns=%s", cfg.id_columns)
            matched = self._join_on_id_columns(
                scored_df,
                actuals_df,
                cfg.id_columns,
                left_name="scored_data",
                right_name="actuals_data",
            )
            if matched.empty:
                raise ValueError(
                    f"No rows matched between scored_data and actuals_data on "
                    f"id_columns={cfg.id_columns!r} — cannot compute monitoring metrics. "
                    "Check that id_columns was set consistently on the original "
                    "ScoringRunner run that produced scored_data."
                )

            logger.info("[5/6] Computing performance metrics (task=%s)", scoring_pipeline.task)
            performance_report = self._compute_performance(
                matched, scoring_pipeline, cfg.actuals_data.target_column
            )

            feature_drift_report = None
            if cfg.raw_data is not None:
                logger.info("[6/6] Computing feature drift (CSI)")
                raw_df = load_raw_data(
                    cfg.raw_data.path, cfg.raw_data.format, cfg.raw_data.sheet_name
                )
                raw_matched = raw_df.merge(
                    matched[cfg.id_columns].drop_duplicates(), on=cfg.id_columns, how="inner"
                )
                if raw_matched.empty:
                    logger.warning(
                        "raw_data join on id_columns matched 0 rows against the scored+actuals "
                        "population — skipping feature drift report."
                    )
                else:
                    feature_drift_report = scoring_pipeline.compute_feature_drift(raw_matched)
            else:
                logger.info("[6/6] Feature drift skipped (raw_data not set)")

            ext = self._ext(cfg.output.format)
            performance_report_path = self._write_report(
                performance_report, run_dir / f"performance_report.{ext}", cfg.output.format
            )
            logger.info("      Performance report: %s", performance_report_path)

            feature_drift_report_path = None
            if feature_drift_report is not None:
                feature_drift_report_path = self._write_report(
                    feature_drift_report, run_dir / f"feature_drift_report.{ext}", cfg.output.format
                )
                logger.info("      Feature drift report: %s", feature_drift_report_path)

            config_path = run_dir / "config.yaml"
            with open(config_path, "w") as fh:
                yaml.dump(cfg.model_dump(), fh, default_flow_style=False, sort_keys=False)

            elapsed = time.time() - t0
            logger.info("=" * 60)
            logger.info("Monitoring complete — %.1fs  |  Run: %s", elapsed, run_id)
            logger.info("=" * 60)

            return MonitoringRunResult(
                config=cfg,
                performance_report=performance_report,
                feature_drift_report=feature_drift_report,
                n_scored_rows=len(scored_df),
                n_actuals_rows=len(actuals_df),
                n_matched_rows=len(matched),
                run_id=run_id,
                run_dir=run_dir,
                performance_report_path=performance_report_path,
                feature_drift_report_path=feature_drift_report_path,
                log_path=log_path,
                elapsed_seconds=elapsed,
            )
        finally:
            detach_run_log_handler(handler, prev_level)

    @staticmethod
    def _join_on_id_columns(
        left: pd.DataFrame,
        right: pd.DataFrame,
        id_columns: list[str],
        left_name: str,
        right_name: str,
    ) -> pd.DataFrame:
        """Inner-join ``left``/``right`` on ``id_columns``, logging a non-fatal warning
        on any row-count mismatch.

        Args:
            left (pd.DataFrame): First frame (e.g. ``scored_data``).
            right (pd.DataFrame): Second frame (e.g. ``actuals_data``).
            id_columns (list[str]): Join key(s).
            left_name (str): Name for ``left``, used only in the log message.
            right_name (str): Name for ``right``, used only in the log
                message.

        Returns:
            pd.DataFrame: The inner-joined result (may be empty).
        """
        merged = left.merge(right, on=id_columns, how="inner", suffixes=("", "_actual"))
        unmatched_left = len(left) - len(merged)
        unmatched_right = len(right) - len(merged)
        if unmatched_left or unmatched_right:
            logger.warning(
                "id_columns join matched %d row(s) — %d %s row(s) had no matching %s row, "
                "%d %s row(s) had no matching %s row (dropped, non-fatal).",
                len(merged),
                unmatched_left,
                left_name,
                right_name,
                unmatched_right,
                right_name,
                left_name,
            )
        return merged

    @staticmethod
    def _compute_performance(
        matched: pd.DataFrame, scoring_pipeline: ScoringPipeline, target_column: str
    ) -> pd.DataFrame:
        """Re-measure performance against actuals using the exact ``_compute_metrics``
        staticmethod training used — raw arrays only, no fitted estimator involved.

        Args:
            matched (pd.DataFrame): Inner join of ``scored_data``/``actuals_data``
                on ``id_columns`` — must contain ``target_column``,
                ``"prediction"``, and (classification) ``"probability"``.
            scoring_pipeline (ScoringPipeline): The loaded bundle — supplies
                ``task`` and ``psi_reference_``.
            target_column (str): Name of the actual-outcome column in
                ``matched``.

        Returns:
            pd.DataFrame: Columns ``"split"`` (always ``"monitoring"``),
            ``"metric"``, ``"value"``.

        Raises:
            ValueError: If ``scoring_pipeline.task`` is ``"clustering"``
                (no ground-truth target to monitor against).
        """
        y_true = matched[target_column].to_numpy()
        y_pred = matched["prediction"].to_numpy()
        y_prob = matched["probability"].to_numpy() if "probability" in matched.columns else None

        if scoring_pipeline.task == "classification":
            train_scores = (
                scoring_pipeline.psi_reference_.to_numpy()
                if scoring_pipeline.psi_reference_ is not None
                else None
            )
            metrics = ClassificationModel._compute_metrics(
                y_true, y_pred, y_prob, split_name="monitoring", train_scores=train_scores
            )
        elif scoring_pipeline.task == "regression":
            metrics = RegressionModel._compute_metrics(
                y_true, y_pred, y_prob, split_name="monitoring"
            )
        else:
            raise ValueError(
                f"MonitoringRunner does not support task={scoring_pipeline.task!r} — "
                "clustering has no ground-truth target to monitor against."
            )

        return pd.DataFrame(
            [
                {
                    "split": "monitoring",
                    "metric": k,
                    "value": round(v, settings.evaluate_round_precision),
                }
                for k, v in metrics.items()
            ]
        )

    @staticmethod
    def _ext(fmt: str) -> str:
        return "xlsx" if fmt == "excel" else fmt

    @staticmethod
    def _write_report(report: pd.DataFrame, path: Path, fmt: str) -> Path:
        """Write a report DataFrame to disk — mirrors ``ScoringRunner._write_output``'s
        format dispatch (parquet/csv/excel).

        Args:
            report (pd.DataFrame): Report to write.
            path (Path): Destination path.
            fmt (str): One of ``"parquet"``, ``"csv"``, ``"excel"``.

        Returns:
            Path: The path written to.
        """
        if fmt == "parquet":
            report.to_parquet(path, index=False)
        elif fmt == "csv":
            report.to_csv(path, index=False)
        else:  # excel
            report.to_excel(path, index=False)
        return path
