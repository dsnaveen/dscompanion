"""PipelineRunner: end-to-end ML experiment executor driven by PipelineConfig."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from dscompanion.config import settings
from dscompanion.pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)

__all__ = ["PipelineRunner", "PipelineRunResult"]

# ── Result container ──────────────────────────────────────────────────────────


@dataclass
class PipelineRunResult:
    """All artefacts produced by a completed pipeline run.

    Returned by ``PipelineRunner.run()``.  Every field is populated before
    the method returns so callers can access any artefact without additional
    calls.

    Args:
        config (PipelineConfig): The full validated config that drove this run.
        split: Fitted ``DataSplit`` after feature processing and selection.
        feature_pipeline: Fitted ``FeatureProcessingPipeline``.
        selection_pipeline: Fitted ``FeatureSelectionPipeline``.
        model: Fitted ``BaseDSCompanionModel`` (tuned if ``tuning.enabled=True``).
        metrics (pd.DataFrame): Per-split evaluation metrics from
            ``model.evaluate()``.
        explainer: Fitted ``SHAPExplainer``, or ``None`` when
            ``explain.shap_enabled=False``.
        permutation_importance: Fitted ``PermutationImportanceAnalyser``, or
            ``None`` when ``explain.permutation_enabled=False``. Independent
            of ``explainer`` — carries none of SHAP's crash risk, so it can
            be used in place of (or alongside) it.
        calibrator: Fitted ``Calibrator``, or ``None`` for regression /
            clustering tasks.
        tuner: ``Tuner`` instance after ``run()``, or ``None`` when
            ``tuning.enabled=False``.
        leaderboard: ``Leaderboard`` instance after ``run()``, or ``None``
            when ``leaderboard.enabled=False``. When set, ``model`` is the
            top-ranked candidate (``leaderboard.best_model()``) and
            ``config.model.algorithm`` has been updated in place to match.
        model_card: Generated ``ModelCard`` instance — ``model_card.to_excel(...)``
            / ``.to_html(...)`` / ``.to_word(...)`` / ``.to_json(...)`` write the report
            to disk on demand.
        run_id (str | None): Local tracking run ID, or ``None`` when
            tracking is unavailable.
        run_dir (Path | None): ``<reporting.output_dir>/<run_id>/`` — the
            root directory this run's artifacts are organized under
            (``model/``, ``reports/``, ``logs/``, ``eda/`` subdirectories).
            ``run_dir.name == run_id`` always holds.
        log_path (Path | None): ``<run_dir>/logs/run.log`` — every INFO+
            log line the ``dscompanion`` logger emitted during this run, captured
            via a temporary ``logging.FileHandler`` attached for the
            duration of ``run()`` and removed afterward (even on failure).
        model_path (Path | None): ``<run_dir>/model/<name>_v<version>_model.joblib``
            — the trained (and tuned/calibrated) model, saved automatically
            via ``model.save()``. ``None`` only if the save itself failed
            (logged as a non-fatal warning; the rest of the run still
            completes) — check ``model_path is not None`` before relying on
            it, don't assume ``model`` was written to disk just because the
            run succeeded.
        report_path (Path | None): Path to the written HTML model report, or
            ``None`` when ``reporting.html_report=False`` (the default) —
            ``model_card.to_excel(...)``/``.to_html(...)``/``.to_json(...)`` remain
            available either way.
        excel_report_path (Path | None): ``<run_dir>/reports/<name>_v<version>_model_card.xlsx``
            — the Excel model card, saved automatically (unlike
            ``report_path``, this is unconditional, not gated behind a
            config flag). ``None`` only if the write itself failed (logged
            as a non-fatal warning).
        config_deviations (list[dict[str, Any]]): One dict per config field set
            to a non-default value, each with keys ``"parameter"``,
            ``"options"``, ``"default"``, and ``"user_choice"`` — surfaced as
            the "Non-Default Choices" callout at the top of the model card.
        elapsed_seconds (float): Wall-clock seconds from ``run()`` start to
            finish.

    Returns:
        PipelineRunResult: Fully populated result container.
    """

    config: PipelineConfig
    split: Any
    feature_pipeline: Any
    selection_pipeline: Any
    model: Any
    metrics: pd.DataFrame
    explainer: Any | None = None
    permutation_importance: Any | None = None
    calibrator: Any | None = None
    tuner: Any | None = None
    leaderboard: Any | None = None
    model_card: Any | None = None
    run_id: str | None = None
    run_dir: Path | None = None
    log_path: Path | None = None
    model_path: Path | None = None
    report_path: Path | None = None
    excel_report_path: Path | None = None
    config_deviations: list[dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0


# ── Runner ────────────────────────────────────────────────────────────────────


class PipelineRunner:
    """Execute a full ML experiment from a ``PipelineConfig`` and return all artefacts.

    Runs 13 stages in fixed order: data loading → splitting → EDA →
    feature processing → feature selection → target treatment →
    model training → optional tuning → evaluation → calibration →
    SHAP → model card → run tracking + report writing.

    Non-default config choices (e.g. ``eda.enabled=False``) are collected and
    surfaced in the model report so the lead reviewer can see what was turned
    off without reading the full YAML.

    Args:
        config (PipelineConfig): Validated pipeline configuration.

    Example::

        result = PipelineRunner.from_yaml("experiments/credit_risk_v1.yaml").run()
        print(result.metrics)
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._deviations: list[str] = []
        self._run_id: str | None = None
        self._run_dir: Path | None = None
        self._prev_dscompanion_level: int = logging.NOTSET

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineRunner":
        """Construct a ``PipelineRunner`` by loading and validating a YAML config file.

        Args:
            path (str | Path): Path to the experiment YAML file.

        Returns:
            PipelineRunner: Ready to call ``.run()`` on.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            pydantic.ValidationError: If the config is invalid.
        """
        return cls(PipelineConfig.from_yaml(path))

    def _generate_run_id_and_dir(self, output_dir: str | Path) -> tuple[str, Path]:
        """Generate a fresh, collision-safe run id and its output directory.

        Uses ``yyyymmdd_hhmmss`` rather than a random uuid — sortable
        chronologically in a Workspace file browser, and directly readable
        as "when did this run happen" without opening it. Only appends a
        short random disambiguating suffix in the rare case two runs start
        in the same second against the same ``output_dir`` (checked via
        directory existence) — the common case stays a clean, readable id.

        Args:
            output_dir (str | Path): Root directory this run's output lives
                under (``cfg.reporting.output_dir``).

        Returns:
            tuple[str, Path]: ``(run_id, run_dir)`` — ``run_dir`` is always
            ``Path(output_dir) / run_id`` and is guaranteed not to already
            exist at the time this returns.
        """
        output_dir = Path(output_dir)
        run_id = time.strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / run_id
        if run_dir.exists():
            run_id = f"{run_id}_{uuid.uuid4().hex[:4]}"
            run_dir = output_dir / run_id
        return run_id, run_dir

    def run(self) -> PipelineRunResult:
        """Execute all pipeline stages and return a fully populated ``PipelineRunResult``.

        Stages run in fixed order regardless of config.  Config flags control
        depth (e.g. ``eda.bivariate=False`` skips bivariate EDA) but cannot
        reorder or skip structural stages.  Logs a timestamped INFO line at
        the start and end of each stage.

        Args:
            None

        Returns:
            PipelineRunResult: All fitted artefacts, metrics, run ID,
            report path, and any config deviation warnings.

        Raises:
            ValueError: If pre-flight cross-config validation fails (e.g.
                ``split.method=temporal`` with no ``data.date_column``).
            RuntimeError: If data loading fails or the target column is absent.
        """
        t0 = time.time()
        cfg = self.config
        logger.info("=" * 60)
        logger.info("PipelineRunner  |  %s  v%s", cfg.name, cfg.version)
        logger.info(
            "Owner: %s  |  Task: %s  |  Algorithm: %s",
            cfg.owner or "unset",
            cfg.model.task,
            cfg.model.algorithm,
        )
        logger.info("=" * 60)

        # Stage 0 — pre-flight
        self._preflight()
        self._deviations = self._detect_config_deviations()
        if self._deviations:
            logger.warning("Non-default config choices (will appear in report):")
            for d in self._deviations:
                logger.warning(
                    "  ⚠  %s = %s  (default: %s)",
                    d["parameter"],
                    d["user_choice"],
                    d["default"],
                )

        # Run identity + output directory — generated once, up front, so the
        # directory can be named after the id and every later stage (model
        # save, report write, log capture) writes into the same tree. The
        # same id is threaded into tracking_run() at the end (_log_and_write)
        # rather than letting it mint a second, different one.
        self._run_id, self._run_dir = self._generate_run_id_and_dir(cfg.reporting.output_dir)
        for subdir in ("model", "reports", "logs", "eda"):
            (self._run_dir / subdir).mkdir(parents=True, exist_ok=True)
        logger.info("Run directory: %s", self._run_dir)

        log_path = self._run_dir / "logs" / "run.log"
        log_handler = self._attach_run_log_handler(log_path)
        try:
            return self._execute_stages(cfg, t0, log_path)
        finally:
            self._detach_run_log_handler(log_handler)

    def _attach_run_log_handler(self, log_path: Path) -> logging.Handler:
        """Attach a ``FileHandler`` to the ``dscompanion`` logger for the duration of a run.

        Also guarantees the ``dscompanion`` logger's effective level is INFO for
        the run's duration — confirmed on Databricks 2026-09-10 that a
        notebook host can pre-install its own root logger handler *before*
        ``import dscompanion`` runs, which silently makes
        ``dscompanion/__init__.py``'s own ``logging.basicConfig(level=INFO)`` a
        no-op (per Python's own docs: ``basicConfig()`` does nothing if the
        root logger already has handlers). Left unguarded, ``dscompanion``'s
        effective level then falls back to root's default ``WARNING``, so
        every ``logger.info(...)`` call in ``dscompanion`` is filtered out
        *before a ``LogRecord`` is even created* — no handler, however
        attached, can capture what was never created. Only raises the
        level when it's currently coarser than INFO; never narrows an
        existing, more verbose setting (e.g. a caller-configured DEBUG).

        Args:
            log_path (Path): Destination file — parent directory must
                already exist (``run()`` creates ``<run_dir>/logs/`` up
                front).

        Returns:
            logging.Handler: The attached handler — pass to
            ``_detach_run_log_handler`` when the run finishes.
        """
        handler = logging.FileHandler(log_path)
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
        )
        dscompanion_logger = logging.getLogger("dscompanion")
        dscompanion_logger.addHandler(handler)
        self._prev_dscompanion_level = dscompanion_logger.level
        if dscompanion_logger.getEffectiveLevel() > logging.INFO:
            dscompanion_logger.setLevel(logging.INFO)
        return handler

    def _detach_run_log_handler(self, handler: logging.Handler) -> None:
        """Remove a handler and restore the level ``_attach_run_log_handler`` set.

        Runs in a ``finally`` block in ``run()`` so both are undone even
        when a stage raises — never leaks across multiple runs in the same
        process/notebook session.

        Args:
            handler (logging.Handler): The handler to remove.

        Returns:
            None
        """
        dscompanion_logger = logging.getLogger("dscompanion")
        dscompanion_logger.removeHandler(handler)
        dscompanion_logger.setLevel(self._prev_dscompanion_level)
        handler.close()

    def _execute_stages(self, cfg: PipelineConfig, t0: float, log_path: Path) -> PipelineRunResult:
        """Run stages 1-13 and assemble the final ``PipelineRunResult``.

        Split out of ``run()`` so the log-capture ``FileHandler`` (attached
        in ``run()``, before this is called) is guaranteed removed via a
        ``finally`` block regardless of how this method exits.

        Args:
            cfg (PipelineConfig): This runner's validated config (same
                object as ``self.config``).
            t0 (float): ``time.time()`` timestamp from the start of
                ``run()``, used to compute ``elapsed_seconds``.
            log_path (Path): Path the run's log file was written to —
                threaded through only to populate
                ``PipelineRunResult.log_path``.

        Returns:
            PipelineRunResult: All fitted artefacts, metrics, run ID,
            report paths, and any config deviation warnings.
        """
        # Stage 1 — load + split
        logger.info("[1/13] Loading data")
        df = self._load_data()
        logger.info("       Loaded %d rows × %d columns", len(df), len(df.columns))

        logger.info("[2/13] Splitting data")
        raw_split = self._split_data(df)

        # Stage 2 — EDA
        logger.info("[3/13] EDA")
        eda_report = self._run_eda(raw_split)

        # Stage 3 — target binarisation (optional)
        logger.info("[4/13] Target treatment")
        raw_split = self._binarize_target(raw_split)

        # Stage 4 — feature processing
        logger.info("[5/13] Feature processing (impute → encode → scale)")
        feat_pipeline = self._build_feature_pipeline()
        processed_split = self._apply_feature_pipeline(feat_pipeline, raw_split)

        # Stage 5 — feature selection
        logger.info("[6/13] Feature selection")
        sel_pipeline = self._build_selection_pipeline()
        selected_split = self._apply_selection_pipeline(sel_pipeline, processed_split)
        n_in = processed_split.train_X.shape[1]
        n_out = selected_split.train_X.shape[1]
        logger.info("       Features: %d → %d (removed %d)", n_in, n_out, n_in - n_out)

        # Stage 6 — imbalance handling
        logger.info("[7/13] Imbalance handling")
        final_split, imbalance_handler = self._handle_imbalance(selected_split)

        # Stage 7 — train (or leaderboard comparison)
        leaderboard = None
        if cfg.leaderboard.enabled:
            logger.info(
                "[8/13] Leaderboard: comparing algorithms (model.algorithm=%s is ignored)",
                cfg.model.algorithm,
            )
            model, leaderboard = self._run_leaderboard(final_split)
            cfg.model.algorithm = leaderboard.best_algorithm()
            logger.info("       Leaderboard winner: %s", cfg.model.algorithm)
        else:
            logger.info("[8/13] Training %s", cfg.model.algorithm)
            model = self._build_model()
            model = self._train_model(model, final_split, imbalance_handler)
            # PSI baseline: when SMOTE/undersample was applied, _train_scores was
            # set on synthetic/inflated data. Re-score on the original pre-resampling
            # training set so PSI measures real population drift, not synthetic drift.
            if (
                imbalance_handler is not None
                and cfg.target.imbalance.strategy not in ("none", "class_weight")
                and hasattr(model, "_train_scores")
                and hasattr(model.estimator, "predict_proba")
            ):
                model._train_scores = model.estimator.predict_proba(selected_split.train_X)[:, 1]
                logger.info(
                    "       PSI baseline re-scored on original (pre-resampling) training set"
                )

        # Stage 8 — tuning (optional)
        tuner = None
        if cfg.tuning.enabled:
            logger.info("[9/13] Hyperparameter tuning (%d trials)", cfg.tuning.n_trials)
            model, tuner = self._tune_model(model, final_split)
        else:
            logger.info("[9/13] Tuning skipped (tuning.enabled=False)")

        # Stage 9 — evaluate
        logger.info("[10/13] Evaluating")
        metrics = model.evaluate(final_split)

        # Stage 10 — calibration
        logger.info("[11/13] Calibration")
        calibrator = self._calibrate(model, final_split)

        # Save the trained model — non-fatal: a save failure shouldn't discard an
        # otherwise-successful run's in-memory artifacts.
        model_path = self._save_model(model)

        # Stage 11 — SHAP
        explainer = None
        if cfg.explain.shap_enabled:
            logger.info("[12/13] SHAP (sample=%d)", cfg.explain.shap_sample_size)
            explainer = self._run_shap(model, final_split)
        else:
            logger.info("[12/13] SHAP skipped (explain.shap_enabled=False)")

        permutation_importance = None
        if cfg.explain.permutation_enabled:
            logger.info(
                "[12/13] Permutation importance (n_repeats=%d, sample=%d)",
                cfg.explain.permutation_n_repeats,
                cfg.explain.permutation_sample_size,
            )
            permutation_importance = self._run_permutation_importance(model, final_split)
        else:
            logger.info(
                "[12/13] Permutation importance skipped (explain.permutation_enabled=False)"
            )

        # Stage 12 — model card + report
        logger.info("[13/13] Generating report + logging run")
        model_card = self._generate_model_card(
            model,
            final_split,
            explainer,
            calibrator,
            tuner,
            eda_report,
            leaderboard,
            permutation_importance,
            eda_split=raw_split,
        )
        run_id, report_path, excel_report_path = self._log_and_write(
            model, final_split, model_card, metrics, feat_pipeline, sel_pipeline
        )

        elapsed = time.time() - t0
        logger.info("=" * 60)
        logger.info("Pipeline complete — %.1fs  |  Run: %s", elapsed, run_id or "n/a")
        logger.info("Report: %s", report_path or "n/a")
        logger.info("=" * 60)

        return PipelineRunResult(
            config=cfg,
            split=final_split,
            feature_pipeline=feat_pipeline,
            selection_pipeline=sel_pipeline,
            model=model,
            metrics=metrics,
            explainer=explainer,
            permutation_importance=permutation_importance,
            calibrator=calibrator,
            tuner=tuner,
            leaderboard=leaderboard,
            model_card=model_card,
            run_id=run_id,
            run_dir=self._run_dir,
            log_path=log_path,
            model_path=model_path,
            report_path=report_path,
            excel_report_path=excel_report_path,
            config_deviations=self._deviations,
            elapsed_seconds=elapsed,
        )

    # ── Pre-flight ────────────────────────────────────────────────────────────

    def _preflight(self) -> None:
        cfg = self.config
        errors: list[str] = []
        warnings: list[str] = []

        # Cross-config structural checks
        if cfg.split.method == "temporal" and not cfg.data.date_column:
            errors.append("split.method='temporal' requires data.date_column to be set.")
        if cfg.split.method == "grouped" and not cfg.split.group_column:
            errors.append("split.method='grouped' requires split.group_column to be set.")

        # Logistic regression without scaling
        if cfg.model.algorithm == "logistic" and cfg.features.scaler.strategy == "none":
            warnings.append(
                "algorithm='logistic' works best with scaling. "
                "Auto-correcting features.scaler.strategy to 'standard'."
            )
            # Mutate in place — pydantic v2 models are mutable by default (not frozen)
            cfg.features.scaler.strategy = "standard"

        # Tuning with too few trials
        if cfg.tuning.enabled and cfg.tuning.n_trials < 10:
            warnings.append(
                f"tuning.n_trials={cfg.tuning.n_trials} is very low. "
                "Results will be unreliable. Recommend >= 50."
            )

        # Imbalance strategy on non-classification
        if cfg.model.task != "classification" and cfg.target.imbalance.strategy != "none":
            warnings.append(
                f"target.imbalance.strategy='{cfg.target.imbalance.strategy}' is only "
                "applicable to classification tasks. It will be ignored."
            )

        # VIF enabled but no selector exists
        if cfg.selection.vif_enabled:
            warnings.append(
                "selection.vif_enabled=True: VIF selector is not yet implemented in "
                "dscompanion. vif_enabled will be ignored this run."
            )

        if errors:
            raise ValueError(
                "PipelineConfig pre-flight failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )
        for w in warnings:
            logger.warning("Pre-flight: %s", w)

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_data(self) -> pd.DataFrame:
        cfg = self.config
        path = cfg.data.path
        fmt = cfg.data.format

        try:
            if fmt == "parquet":
                df = pd.read_parquet(path)
            elif fmt == "csv":
                df = pd.read_csv(path)
            elif fmt == "excel":
                df = self._load_excel(path, cfg.data.sheet_name)
            elif fmt == "delta":
                df = self._load_delta(path)
            else:
                raise ValueError(f"Unsupported format: {fmt!r}")
        except Exception as exc:
            raise RuntimeError(f"Failed to load data from {path!r}: {exc}") from exc

        # Dev-only row subsampling — random, not "first N", so a small sample stays
        # representative rather than biased toward however the source file is
        # ordered. Seeded via settings.random_state for reproducibility. Applied
        # uniformly across every format (CSV used to take a native, sequential
        # read_csv(nrows=...) shortcut that avoided loading the full file — traded
        # away here for consistent random-sampling behavior across all formats).
        if cfg.data.nrows:
            df = df.sample(n=min(cfg.data.nrows, len(df)), random_state=settings.random_state)
        elif cfg.data.fraction_rows:
            df = df.sample(frac=cfg.data.fraction_rows, random_state=settings.random_state)

        if cfg.data.target not in df.columns:
            raise RuntimeError(
                f"Target column '{cfg.data.target}' not found in dataset. "
                f"Available columns: {list(df.columns[:10])}..."
            )

        # Restrict to selected features + target (+ date column if needed)
        if cfg.data.feature_columns:
            keep = set(cfg.data.feature_columns) | {cfg.data.target}
            if cfg.data.date_column:
                keep.add(cfg.data.date_column)
            missing = keep - set(df.columns)
            if missing:
                raise RuntimeError(f"feature_columns references columns not in dataset: {missing}")
            df = df[sorted(keep)]
        elif cfg.data.ignore_columns:
            missing = set(cfg.data.ignore_columns) - set(df.columns)
            if missing:
                raise RuntimeError(f"ignore_columns references columns not in dataset: {missing}")
            df = df.drop(columns=cfg.data.ignore_columns)

        return df

    def _load_excel(
        self, path: str, sheet_name: str | int | list[str | int] | None
    ) -> pd.DataFrame:
        """Load one Excel file, optionally combining several named/indexed sheets.

        Args:
            path (str): Path to the ``.xlsx``/``.xls`` file.
            sheet_name (str | int | list[str | int] | None): A single sheet
                reads directly; a list reads each sheet and vertically
                concatenates them (every sheet must have identical columns);
                ``None`` reads the first sheet (index ``0``), not every sheet
                in the workbook — matching every other ``format`` here
                reading exactly one dataset from one ``path``.

        Returns:
            pd.DataFrame: The loaded (and, for a list of sheets, concatenated)
            data.

        Raises:
            ValueError: If a list of sheets is given and their column sets
                don't all match.
        """
        result = pd.read_excel(path, sheet_name=0 if sheet_name is None else sheet_name)
        if isinstance(result, dict):
            frames = list(result.items())
            first_name, first_df = frames[0]
            for name, df in frames[1:]:
                if set(df.columns) != set(first_df.columns):
                    raise ValueError(
                        f"sheet_name list requires identical columns across sheets — "
                        f"sheet {name!r} has columns {sorted(df.columns)}, but sheet "
                        f"{first_name!r} has {sorted(first_df.columns)}."
                    )
            return pd.concat([df for _, df in frames], ignore_index=True)
        return result

    def _load_delta(self, path: str) -> pd.DataFrame:
        try:
            from pyspark.sql import SparkSession

            spark = SparkSession.getActiveSession()
            if spark is None:
                raise RuntimeError("No active SparkSession found.")
            return spark.read.format("delta").load(path).toPandas()
        except ImportError:
            raise RuntimeError(
                "format='delta' requires PySpark. " "Use format='parquet' for local development."
            )

    # ── Splitting ─────────────────────────────────────────────────────────────

    def _split_data(self, df: pd.DataFrame):
        from dscompanion.split import DataSplitter

        cfg = self.config
        splitter = DataSplitter(
            strategy=cfg.split.method,
            test_size=cfg.split.test_size,
            val_size=cfg.split.val_size,
            date_col=cfg.data.date_column,
            group_col=cfg.split.group_column,
            target_col=cfg.data.target,
        )
        return splitter.fit_split(df)

    # ── EDA ───────────────────────────────────────────────────────────────────

    def _run_eda(self, split) -> Any | None:
        cfg = self.config
        if not cfg.eda.enabled:
            logger.info("       EDA disabled — skipping")
            return None

        try:
            from dscompanion.eda import EDAReport

            report = EDAReport(
                split=split,
                target=cfg.data.target,
                high_missing_threshold=cfg.eda.high_missing_threshold,
                near_zero_variance_threshold=cfg.eda.near_zero_variance_threshold,
                skewness_alert_threshold=cfg.eda.skewness_alert_threshold,
                zero_pct_alert_threshold=cfg.eda.zero_pct_alert_threshold,
                imbalance_alert_threshold=cfg.eda.imbalance_alert_threshold,
                extreme_values_n=cfg.eda.extreme_values_n,
                sample_n_rows=cfg.eda.sample_n_rows,
                duplicate_rows_max_display=cfg.eda.duplicate_rows_max_display,
                missing_matrix_max_rows=cfg.eda.missing_matrix_max_rows,
                interaction_max_numeric_cols=cfg.eda.interaction_max_numeric_cols,
                interaction_hexbin_row_threshold=cfg.eda.interaction_hexbin_row_threshold,
                text_analysis_top_n_words=cfg.eda.text_analysis_top_n_words,
                text_analysis_top_n_chars=cfg.eda.text_analysis_top_n_chars,
                include_sample_rows=cfg.eda.include_sample_rows,
                include_duplicate_row_content=cfg.eda.include_duplicate_row_content,
                include_text_sample_values=cfg.eda.include_text_sample_values,
            )
            report.run_all()
            logger.info("       EDA complete")
            return report
        except Exception as exc:
            logger.warning("EDA failed (non-fatal): %s", exc)
            return None

    # ── Target binarisation ───────────────────────────────────────────────────

    def _binarize_target(self, split):
        cfg = self.config
        threshold = cfg.target.binarize_threshold
        if threshold is None:
            return split

        from dscompanion.targets import TargetBinariser

        binariser = TargetBinariser(strategy="threshold", threshold=threshold)
        binariser.fit(split.train_y)

        return self._rebuild_split(
            split,
            {
                "train_y": binariser.transform(split.train_y),
                "val_y": binariser.transform(split.val_y),
                "test_y": binariser.transform(split.test_y),
                "oot_y": binariser.transform(split.oot_y) if len(split.oot_y) > 0 else split.oot_y,
            },
            update_y=True,
        )

    # ── Feature processing ────────────────────────────────────────────────────

    def _build_feature_pipeline(self):
        from dscompanion.features import (
            FeatureProcessingPipeline,
            FeatureTransformChain,
            SmartImputer,
        )

        cfg = self.config

        if cfg.feature_recipes is not None:
            logger.info(
                "Recipe-based feature processing (%d columns) — note: FeatureTransformChain "
                "does not run LeakageGuard automatically, unlike FeatureProcessingPipeline's "
                "run_leakage_check; add a 'target_encode'/'woe_encode' step's own leakage "
                "handling per column or run LeakageGuard separately if needed.",
                len(cfg.feature_recipes),
            )
            return FeatureTransformChain(recipes=cfg.feature_recipes)

        encoder_map = {"ordinal": "ordinal", "target": "woe", "onehot": "onehot"}
        encoder = encoder_map.get(cfg.features.encoder.strategy, "ordinal")

        imputer = SmartImputer(
            numeric_strategy=cfg.features.imputer.numeric_strategy,
            categorical_strategy=cfg.features.imputer.categorical_strategy,
            fill_value=cfg.features.imputer.fill_value,
            add_missing_indicator=cfg.features.imputer.add_missing_indicator,
            column_strategies=cfg.features.imputer.column_strategies,
            column_fill_values=cfg.features.imputer.column_fill_values,
        )

        return FeatureProcessingPipeline(
            imputer=imputer,
            encoder=encoder,
            scaler=cfg.features.scaler.strategy,
            distribution=cfg.features.distribution.strategy,
            run_leakage_check=cfg.selection.leakage_check,
            winsorize=cfg.features.winsorizer.enabled,
            winsorize_lower=cfg.features.winsorizer.lower_tail,
            winsorize_upper=cfg.features.winsorizer.upper_tail,
        )

    def _apply_feature_pipeline(self, feat_pipeline, split):
        train_X = feat_pipeline.fit_transform(split.train_X, split.train_y)
        new_X = {"train_X": train_X}

        for name, attr in [
            ("val_X", split.val_X),
            ("test_X", split.test_X),
            ("oot_X", split.oot_X),
        ]:
            if len(attr) > 0:
                new_X[name] = feat_pipeline.transform(attr)
            else:
                new_X[name] = attr

        return self._rebuild_split(split, new_X)

    # ── Feature selection ─────────────────────────────────────────────────────

    def _build_selection_pipeline(self):
        from dscompanion.selection import FeatureSelectionPipeline
        from dscompanion.selection.feature_selectors import (
            CardinalitySelector,
            ConstantSelector,
            CorrelationSelector,
            IVSelector,
            NullRateSelector,
        )

        cfg = self.config
        sel = cfg.selection
        selectors = []

        if sel.remove_high_null:
            selectors.append(NullRateSelector(threshold=sel.null_rate_threshold))

        if sel.remove_constant or sel.remove_quasi_constant:
            selectors.append(ConstantSelector(threshold=sel.quasi_constant_threshold))

        if sel.remove_high_cardinality:
            selectors.append(CardinalitySelector(max_cardinality=sel.cardinality_threshold))

        if sel.remove_high_correlation:
            selectors.append(CorrelationSelector(threshold=sel.correlation_threshold))

        # IV selector only for classification
        if cfg.model.task == "classification":
            selectors.append(IVSelector(threshold=sel.iv_threshold))

        return FeatureSelectionPipeline(selectors=selectors)

    def _apply_selection_pipeline(self, sel_pipeline, split):
        train_X = sel_pipeline.fit_transform(split.train_X, split.train_y)
        new_X = {"train_X": train_X}

        for name, attr in [
            ("val_X", split.val_X),
            ("test_X", split.test_X),
            ("oot_X", split.oot_X),
        ]:
            if len(attr) > 0:
                new_X[name] = sel_pipeline.transform(attr)
            else:
                new_X[name] = attr

        return self._rebuild_split(split, new_X)

    # ── Imbalance handling ────────────────────────────────────────────────────

    def _handle_imbalance(self, split) -> tuple[Any, Any | None]:
        cfg = self.config
        strategy = cfg.target.imbalance.strategy

        if cfg.model.task != "classification" or strategy == "none":
            return split, None

        from dscompanion.targets import ImbalanceHandler

        handler = ImbalanceHandler(
            strategy=strategy,
            sampling_strategy=cfg.target.imbalance.sampling_strategy,
        )
        new_train_X, new_train_y = handler.fit_resample(split.train_X, split.train_y)

        new_split = self._rebuild_split(
            split, {"train_X": new_train_X, "train_y": new_train_y}, update_y=True
        )
        return new_split, handler

    # ── Model build + train ───────────────────────────────────────────────────

    def _build_model(self):
        """Build an untrained model wrapper for the configured task and algorithm.

        Args:
            None

        Returns:
            BaseDSCompanionModel: A freshly constructed, unfitted model wrapper
            (``ClassificationModel``, ``RegressionModel``, or
            ``ClusteringModel``, per ``config.model.task``) for
            ``config.model.algorithm``, with ``config.model.params`` merged
            over the algorithm's built-in defaults.
        """
        from dscompanion.models import ModelFactory

        cfg = self.config
        return ModelFactory.build(
            task=cfg.model.task,
            algorithm=cfg.model.algorithm,
            params=cfg.model.params or {},
        )

    def _train_model(self, model, split, imbalance_handler=None):
        # If class_weight strategy, pass weights from imbalance handler
        if (
            imbalance_handler is not None
            and hasattr(imbalance_handler, "class_weights_")
            and imbalance_handler.class_weights_ is not None
        ):
            try:
                model.estimator.set_params(class_weight=imbalance_handler.class_weights_)
            except Exception:
                pass  # estimator does not support class_weight

        # Algorithms with built-in early stopping (e.g. XGBoost's default
        # early_stopping_rounds=20) raise "Must have at least 1 validation dataset
        # for early stopping" if fit without an eval_set at all — independent of
        # which split downstream evaluation/reporting uses.
        if split.val_X is not None and len(split.val_X) > 0:
            eval_set = [(split.val_X, split.val_y)]
        elif split.test_X is not None and len(split.test_X) > 0:
            eval_set = [(split.test_X, split.test_y)]
        else:
            eval_set = None

        model.fit(split.train_X, split.train_y, eval_set=eval_set)
        return model

    # ── Leaderboard ───────────────────────────────────────────────────────────

    def _run_leaderboard(self, split) -> tuple[Any, Any]:
        from dscompanion.leaderboard import Leaderboard

        cfg = self.config
        leaderboard = Leaderboard(
            include=cfg.leaderboard.include,
            exclude=cfg.leaderboard.exclude,
            sort_metric=cfg.leaderboard.sort_metric,
            eval_split=cfg.leaderboard.eval_split,
        )
        leaderboard.run(split)
        return leaderboard.best_model(), leaderboard

    # ── Tuning ────────────────────────────────────────────────────────────────

    def _tune_model(self, model, split) -> tuple[Any, Any]:
        """Run hyperparameter tuning and return the best model found.

        Args:
            model: Untrained (or freshly built) ``BaseDSCompanionModel`` wrapper
                to tune.
            split: ``DataSplit`` supplying the cross-validation folds/data
                each trial is evaluated against.

        Returns:
            tuple[Any, Any]: ``(best_model, tuner)`` — the best-scoring
            fitted model found across ``config.tuning.n_trials`` trials, and
            the ``Tuner`` instance itself (exposed on
            ``PipelineRunResult.tuner`` for trial-history inspection).
        """
        from dscompanion.tuning import Tuner

        cfg = self.config
        tuner = Tuner(
            model=model,
            backend=cfg.tuning.backend,
            n_trials=cfg.tuning.n_trials,
            cv=split,
            metric=cfg.tuning.metric,
            direction=cfg.tuning.direction,
        )
        best_model = tuner.run()
        return best_model, tuner

    # ── Calibration ───────────────────────────────────────────────────────────

    def _calibrate(self, model, split) -> Any | None:
        if self.config.model.task != "classification":
            return None

        try:
            from dscompanion.calibration import Calibrator

            cal = Calibrator(method="isotonic")
            cal.fit(model, split.val_X, split.val_y)
            return cal
        except Exception as exc:
            logger.warning("Calibration failed (non-fatal): %s", exc)
            return None

    # ── Model persistence ────────────────────────────────────────────────────

    def _save_model(self, model) -> Path | None:
        """Save the trained model into ``<run_dir>/model/`` via ``model.save()``.

        Non-fatal: a save failure is logged as a warning, not raised —
        matches ``_run_shap``/``_run_permutation_importance``'s existing
        defensive shape, since a persistence failure shouldn't discard an
        otherwise-successful run's in-memory artifacts.

        Args:
            model: Fitted ``BaseDSCompanionModel`` to persist.

        Returns:
            Path | None: The resolved path the model was saved to, or
            ``None`` if the save failed.
        """
        cfg = self.config
        path = self._run_dir / "model" / f"{cfg.name}_v{cfg.version}_model.joblib"
        try:
            return model.save(path)
        except Exception as exc:
            logger.warning("Model save failed (non-fatal): %s", exc)
            return None

    # ── SHAP ──────────────────────────────────────────────────────────────────

    def _run_shap(self, model, split) -> Any | None:
        try:
            from dscompanion.explain import SHAPExplainer

            cfg = self.config
            sample_size = min(cfg.explain.shap_sample_size, len(split.test_X))
            X_sample = split.test_X.sample(n=sample_size, random_state=settings.random_state)

            explainer = SHAPExplainer(
                model=model,
                max_display=cfg.explain.shap_top_n,
            )
            explainer.fit(X_sample)
            return explainer
        except Exception as exc:
            logger.warning("SHAP failed (non-fatal): %s", exc)
            return None

    # ── Permutation importance ───────────────────────────────────────────────

    def _run_permutation_importance(self, model, split) -> Any | None:
        try:
            from dscompanion.explain import PermutationImportanceAnalyser

            cfg = self.config
            sample_size = min(cfg.explain.permutation_sample_size, len(split.test_X))
            idx = split.test_X.sample(n=sample_size, random_state=settings.random_state).index
            X_sample = split.test_X.loc[idx]
            y_sample = split.test_y.loc[idx]

            scoring = "roc_auc" if cfg.model.task == "classification" else "r2"
            analyser = PermutationImportanceAnalyser(
                model=model,
                scoring=scoring,
                n_repeats=cfg.explain.permutation_n_repeats,
                top_n=cfg.explain.permutation_top_n,
            )
            analyser.fit(X_sample, y_sample)
            return analyser
        except Exception as exc:
            logger.warning("Permutation importance failed (non-fatal): %s", exc)
            return None

    # ── Model card ────────────────────────────────────────────────────────────

    def _generate_model_card(
        self,
        model,
        split,
        explainer,
        calibrator,
        tuner,
        eda_report,
        leaderboard=None,
        permutation_importance=None,
        eda_split=None,
    ) -> Any:
        """Build and populate the ``ModelCard`` for this run.

        Args:
            model: Fitted ``BaseDSCompanionModel`` wrapper.
            split: Model-ready ``DataSplit`` (post impute/encode/scale/
                select/imbalance) — used for anything that calls into the
                model (metrics, decile table, PSI, leaderboard per-split
                metrics).
            explainer: Fitted SHAP explainer, or ``None``.
            calibrator: Fitted ``Calibrator``, or ``None``.
            tuner: ``Tuner`` instance from hyperparameter search, or
                ``None``.
            eda_report: ``EDAReport`` from the EDA stage, or ``None``.
            leaderboard (optional): Fitted ``Leaderboard`` instance.
            permutation_importance (optional): Fitted permutation-
                importance analyser.
            eda_split (optional): Pre-feature-engineering ``DataSplit``
                (original dtypes, categorical columns intact) for
                ``ModelCard``'s per-split EDA sections (Numeric/
                Categorical/Missing/Categorical Charts/Interactions) —
                those need human-readable raw values, not the
                imputed/encoded/scaled ``split``. Defaults to ``None``,
                which makes ``ModelCard`` fall back to ``split`` itself.

        Returns:
            Any: The generated ``ModelCard`` (``.generate()`` already
            called).
        """
        import yaml as _yaml

        from dscompanion.docs import ModelCard

        cfg = self.config
        raw_config_yaml = _yaml.dump(
            cfg.model_dump(), default_flow_style=False, sort_keys=False, allow_unicode=True
        )
        card = ModelCard(
            model=model,
            split=split,
            eda_split=eda_split,
            explainer=explainer,
            permutation_importance=permutation_importance,
            calibrator=calibrator,
            eda_report=eda_report,
            tuner=tuner,
            leaderboard=leaderboard,
            author=cfg.owner,
            model_version=cfg.version,
            use_case=cfg.description,
            config_checks=self._build_full_config_table(),
            decile_table=cfg.reporting.decile_table,
            raw_config_yaml=raw_config_yaml,
            excel_charts=cfg.eda.chart,
            excel_chart_clip_lower_pct=cfg.eda.chart_clip_lower_pct,
            excel_chart_clip_upper_pct=cfg.eda.chart_clip_upper_pct,
        )
        card.generate()
        return card

    # ── Run tracking + report writing ─────────────────────────────────────────

    def _log_and_write(
        self,
        model,
        split,
        model_card,
        metrics: pd.DataFrame,
        feat_pipeline,
        sel_pipeline,
    ) -> tuple[str | None, Path | None, Path | None]:
        """Open a tracking run, log its artefacts/metrics/params, and write the HTML report.

        Opens a local tracking run via ``tracking_run`` (logs a start/finish
        line via stdlib logging; no network call), logs the resolved
        pipeline config as a YAML artifact, batches the evaluation metrics
        plus before/after feature counts into one ``log_metrics`` call,
        logs the fitted estimator's hyperparameters via ``log_params``, and
        logs any non-default config choices at INFO level. When
        ``config.reporting.html_report=True``, also writes the HTML
        model-card report under ``config.reporting.output_dir`` and logs it
        as an artifact.

        Args:
            model: Fitted ``BaseDSCompanionModel`` whose
                ``estimator.get_params()`` is logged as run parameters.
            split: Fitted ``DataSplit`` — used for
                ``split.train_X.shape[1]`` (post-selection feature count).
            model_card: Generated ``ModelCard`` instance, written to HTML
                when ``config.reporting.html_report=True``.
            metrics (pd.DataFrame): Per-split evaluation metrics with
                ``split``, ``metric``, ``value`` columns, logged as one
                metric per ``{split}_{metric}`` key.
            feat_pipeline: Fitted ``FeatureProcessingPipeline`` — used for
                ``get_feature_names_out()`` (pre-selection feature count)
                when available.
            sel_pipeline: Fitted ``FeatureSelectionPipeline``. Currently
                unused by this method; accepted for call-site symmetry with
                the other post-training stages.

        Returns:
            tuple[str | None, Path | None, Path | None]: ``(run_id, report_path,
            excel_report_path)`` — the tracking run's ``run_id`` (``self._run_id``,
            threaded through via ``tracking_run(run_id=...)``), the HTML
            report path (``None`` when ``config.reporting.html_report=False``),
            and the Excel model card path (unconditional — ``None`` only if
            the write itself failed).
        """
        import yaml as _yaml

        from dscompanion.tracking import log_artifact, log_metrics, log_params, tracking_run

        cfg = self.config
        reports_dir = self._run_dir / "reports"

        run_id = None
        report_path = None
        excel_report_path = None

        with tracking_run(
            run_name=f"{cfg.name}_v{cfg.version}",
            tags={"owner": cfg.owner, "algorithm": cfg.model.algorithm, "task": cfg.model.task},
            run_id=self._run_id,
        ) as run:
            run_id = getattr(getattr(run, "info", None), "run_id", None)

            # Save the resolved config as a real, discoverable file next to the model/report it
            # produced — not a throwaway tempfile deleted right after being logged.
            config_path = self._run_dir / "config.yaml"
            with open(config_path, "w") as fh:
                _yaml.dump(cfg.model_dump(), fh, default_flow_style=False, sort_keys=False)

            try:
                log_artifact(str(config_path), artifact_path="config")

                # Log metrics
                run_metrics: dict[str, float] = {}
                for _, row in metrics.iterrows():
                    key = f"{row.get('split', 'unknown')}_{row['metric']}"
                    try:
                        run_metrics[key] = float(row["value"])
                    except (TypeError, ValueError):
                        pass

                # Log feature counts
                run_metrics["features_before_selection"] = (
                    feat_pipeline.get_feature_names_out().__len__()
                    if hasattr(feat_pipeline, "get_feature_names_out")
                    else -1
                )
                run_metrics["features_after_selection"] = split.train_X.shape[1]
                log_metrics(run_metrics)

                # Log model params
                log_params({k: str(v)[:250] for k, v in model.estimator.get_params().items()})

                # Log config deviations
                if self._deviations:
                    tag_str = " | ".join(
                        f"{d['parameter']}={d['user_choice']} (default {d['default']})"
                        for d in self._deviations
                    )
                    logger.info("config_deviations: %s", tag_str)

            except Exception as exc:
                logger.warning("Run logging failed (non-fatal): %s", exc)

            # Write Excel model card — unconditional, unlike the HTML report below
            try:
                excel_report_path = reports_dir / f"{cfg.name}_v{cfg.version}_model_card.xlsx"
                model_card.to_excel(excel_report_path)
                log_artifact(str(excel_report_path), artifact_path="excel_report")
                logger.info("Excel model card written → %s", excel_report_path)
            except Exception as exc:
                logger.warning("Excel model card write failed (non-fatal): %s", exc)
                excel_report_path = None

            # Write HTML report
            if cfg.reporting.html_report:
                try:
                    report_path = reports_dir / f"{cfg.name}_v{cfg.version}_model_card.html"
                    model_card.to_html(report_path)
                    log_artifact(str(report_path), artifact_path="report")
                    logger.info("HTML report written → %s", report_path)
                except Exception as exc:
                    logger.warning("HTML report write failed (non-fatal): %s", exc)

        return run_id, report_path, excel_report_path

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _rebuild_split(self, old_split, updates: dict[str, Any], update_y: bool = False):
        """Return a new DataSplit replacing the keys in ``updates``."""
        from dscompanion.split import DataSplit

        new_train_X = updates.get("train_X", old_split.train_X)
        new_train_y = updates.get("train_y", old_split.train_y) if update_y else old_split.train_y
        new_val_X = updates.get("val_X", old_split.val_X)
        new_val_y = updates.get("val_y", old_split.val_y) if update_y else old_split.val_y
        new_test_X = updates.get("test_X", old_split.test_X)
        new_test_y = updates.get("test_y", old_split.test_y) if update_y else old_split.test_y
        new_oot_X = updates.get("oot_X", old_split.oot_X)
        new_oot_y = updates.get("oot_y", old_split.oot_y) if update_y else old_split.oot_y

        # Update metadata feature count if X changed
        new_meta = old_split.metadata.model_copy(update={"n_features": new_train_X.shape[1]})

        return DataSplit(
            train_X=new_train_X,
            train_y=new_train_y,
            val_X=new_val_X,
            val_y=new_val_y,
            test_X=new_test_X,
            test_y=new_test_y,
            oot_X=new_oot_X,
            oot_y=new_oot_y,
            metadata=new_meta,
        )

    def _config_checks_spec(self) -> list[tuple[str, Any, Any, str]]:
        """Single source of truth: every monitored ``(parameter, actual, default, choices)`` tuple.

        Covers every behavioural choice surface in ``PipelineConfig`` — stage
        on/off switches, strategy selections (imputer/encoder/scaler/imbalance/
        algorithm/tuning backend), and quality gates — not just a curated
        subset of booleans. Identity fields (``name``, ``version``,
        ``description``, ``owner``), ``data.*``, and ``split.*`` are excluded
        since those are inherently experiment-specific, not "default vs
        override" choices. Numeric thresholds (e.g. ``iv_threshold``,
        ``correlation_threshold``) are excluded too — those are expected,
        routine tuning, not governance-relevant deviations.

        Args:
            None

        Returns:
            list[tuple[str, Any, Any, str]]: One tuple per monitored
            parameter: ``(parameter, actual_value, default_value,
            choices_description)``.
        """
        from dscompanion.models import ModelFactory

        cfg = self.config
        algorithm_choices = " / ".join(ModelFactory.SUPPORTED_ALGORITHMS.get(cfg.model.task, []))

        # When feature_recipes is set, _build_feature_pipeline() returns a
        # FeatureTransformChain and never reads features.imputer/encoder/
        # scaler/distribution at all — showing those 4 sub-sections' values
        # here would misleadingly imply they still governed this run.
        if cfg.feature_recipes is not None:
            feature_rows: list[tuple[str, Any, Any, str]] = [
                (
                    "feature_recipes",
                    f"{len(cfg.feature_recipes)} column(s) with a ColumnRecipe",
                    None,
                    "dict[str, ColumnRecipe] — replaces features.imputer/encoder/scaler/"
                    "distribution entirely for this run",
                ),
            ]
        else:
            feature_rows = [
                (
                    "features.imputer.numeric_strategy",
                    cfg.features.imputer.numeric_strategy,
                    "auto",
                    "auto / mean / median / constant",
                ),
                (
                    "features.imputer.categorical_strategy",
                    cfg.features.imputer.categorical_strategy,
                    "most_frequent",
                    "most_frequent / constant",
                ),
                (
                    "features.imputer.add_missing_indicator",
                    cfg.features.imputer.add_missing_indicator,
                    False,
                    "True / False",
                ),
                (
                    "features.encoder.strategy",
                    cfg.features.encoder.strategy,
                    "ordinal",
                    "ordinal / onehot / target",
                ),
                (
                    "features.scaler.strategy",
                    cfg.features.scaler.strategy,
                    "none",
                    "none / standard / minmax / robust",
                ),
                (
                    "features.distribution.strategy",
                    cfg.features.distribution.strategy,
                    "none",
                    "none / log / log1p / yeo_johnson",
                ),
                (
                    "features.winsorizer.enabled",
                    cfg.features.winsorizer.enabled,
                    False,
                    "True / False",
                ),
            ]

        return [
            ("eda.enabled", cfg.eda.enabled, True, "True / False"),
            ("eda.univariate", cfg.eda.univariate, True, "True / False"),
            ("eda.bivariate", cfg.eda.bivariate, True, "True / False"),
            ("eda.multivariate", cfg.eda.multivariate, False, "True / False"),
            ("eda.chart", cfg.eda.chart, True, "True / False"),
            *feature_rows,
            (
                "target.imbalance.strategy",
                cfg.target.imbalance.strategy,
                "class_weight",
                "class_weight / smote / undersample / oversample / none",
            ),
            ("target.binarize_threshold", cfg.target.binarize_threshold, None, "float / None"),
            ("selection.remove_high_null", cfg.selection.remove_high_null, True, "True / False"),
            ("selection.remove_constant", cfg.selection.remove_constant, True, "True / False"),
            (
                "selection.remove_quasi_constant",
                cfg.selection.remove_quasi_constant,
                True,
                "True / False",
            ),
            (
                "selection.remove_high_cardinality",
                cfg.selection.remove_high_cardinality,
                True,
                "True / False",
            ),
            ("selection.leakage_check", cfg.selection.leakage_check, True, "True / False"),
            (
                "selection.remove_high_correlation",
                cfg.selection.remove_high_correlation,
                True,
                "True / False",
            ),
            ("selection.vif_enabled", cfg.selection.vif_enabled, False, "True / False"),
            ("model.algorithm", cfg.model.algorithm, "xgboost", algorithm_choices),
            ("model.params", cfg.model.params, {}, "dict of estimator-param overrides"),
            ("tuning.enabled", cfg.tuning.enabled, False, "True / False"),
            ("tuning.backend", cfg.tuning.backend, "optuna", "optuna"),
            ("tuning.n_trials", cfg.tuning.n_trials, 50, "int"),
            ("tuning.metric", cfg.tuning.metric, "roc_auc", "any classification metric"),
            ("tuning.direction", cfg.tuning.direction, "maximize", "maximize / minimize"),
            ("leaderboard.enabled", cfg.leaderboard.enabled, False, "True / False"),
            (
                "leaderboard.eval_split",
                cfg.leaderboard.eval_split,
                None,
                "train / val / test / oot / None (auto: val if present, else test)",
            ),
            (
                "leaderboard.sort_metric",
                cfg.leaderboard.sort_metric,
                None,
                "any classification metric / None (auto: settings.classification_metrics[0])",
            ),
            ("explain.shap_enabled", cfg.explain.shap_enabled, False, "True / False"),
            ("explain.lime_enabled", cfg.explain.lime_enabled, False, "True / False"),
            (
                "explain.permutation_enabled",
                cfg.explain.permutation_enabled,
                False,
                "True / False",
            ),
            ("reporting.html_report", cfg.reporting.html_report, False, "True / False"),
            ("reporting.decile_table", cfg.reporting.decile_table, True, "True / False"),
        ]

    def _detect_config_deviations(self) -> list[dict[str, Any]]:
        """Return only the monitored config fields set to non-default values.

        Used for the startup warning log and the ``config_deviations`` log
        line — both want a terse "what changed" list, not the full parameter
        table. See ``_build_full_config_table`` for the complete report-table
        view (every monitored parameter, deviation or not).

        Args:
            None

        Returns:
            list[dict[str, Any]]: One dict per deviation, each with keys
            ``"parameter"`` (str), ``"options"`` (str, the field's valid
            choices), ``"default"`` (the default value), and
            ``"user_choice"`` (the value actually configured for this run).
            Empty list when every monitored field is at its default.
        """
        return [
            {"parameter": parameter, "options": choices, "default": default, "user_choice": actual}
            for parameter, actual, default, choices in self._config_checks_spec()
            if actual != default
        ]

    def _build_full_config_table(self) -> list[dict[str, Any]]:
        """Return every monitored config parameter, deviation or not, for the report table.

        Unlike ``_detect_config_deviations`` (filtered to deviations only),
        this is the complete table a reviewer needs to verify what was
        actually run — Parameter | Choices | Default | User Choice | Flag —
        so "no deviations found" can be confirmed by inspection rather than
        taken on faith.

        Args:
            None

        Returns:
            list[dict[str, Any]]: One dict per monitored parameter, each with
            keys ``"parameter"`` (str), ``"choices"`` (str), ``"default"``,
            ``"user_choice"``, and ``"is_deviation"`` (bool).
        """
        return [
            {
                "parameter": parameter,
                "choices": choices,
                "default": default,
                "user_choice": actual,
                "is_deviation": actual != default,
            }
            for parameter, actual, default, choices in self._config_checks_spec()
        ]
