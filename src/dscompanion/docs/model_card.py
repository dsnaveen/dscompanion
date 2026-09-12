"""ModelCard: auto-generated governance-ready model documentation."""

from __future__ import annotations

import html as _html_lib
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import pandas as pd

from dscompanion.config import settings
from dscompanion.docs import html_eda, html_model
from dscompanion.docs import html_widgets as hw

__all__ = ["ModelCard"]

_SECTION_KEYS = [
    "config_summary",
    "model_summary",
    "data_lineage",
    "feature_inventory",
    "performance_metrics",
    "decile_table",
    "stability",
    "calibration",
    "explainability",
    "hyperparameter_tuning",
    "leaderboard",
    "limitations",
    "governance",
    "full_config",
]

_CONFIG_CHECK_COLUMNS = ["parameter", "choices", "default", "user_choice", "is_deviation"]

# Excel worksheet name platform limit (Excel/xlsxwriter), not a tunable business
# threshold — a code constant, unlike the chart-sizing/threshold values below.
_EXCEL_SHEET_NAME_MAX_LEN = 31
_EXCEL_FORBIDDEN_SHEET_CHARS = frozenset(":\\/?*[]")

# Cosmetic colour mapping for the correlation-matrix conditional-format
# heatmap (styling, not a business threshold, so a module constant rather
# than a settings field).
_EXCEL_HEATMAP_COLORS = {"min": "#c0392b", "mid": "#ffffff", "max": "#2980b9"}

# Packages whose exact version matters most for reproducing or debugging a
# model card later (training libraries + the report-generation stack) — not
# an exhaustive environment dump, just the ones a version mismatch between
# machines is most likely to affect.
_KEY_PACKAGES_FOR_VERSION_REPORT = [
    "pandas",
    "numpy",
    "scikit-learn",
    "scipy",
    "xgboost",
    "lightgbm",
    "catboost",
    "shap",
    "optuna",
    "matplotlib",
    "seaborn",
    "xlsxwriter",
]

# Every Excel worksheet reserves its first row for a "Back to Index" nav
# link, pushing the actual table/header down by this many rows, and its
# first column as a blank visual margin, pushing content right by this many
# columns — together, every sheet's content starts at cell B2, not A1.
_EXCEL_NAV_ROW_OFFSET = 1
_EXCEL_NAV_COL_OFFSET = 1
_EXCEL_NAV_LINK_TEXT = "Back to Index"
_EXCEL_INDEX_SHEET_NAME = "Index"

# One-line description per worksheet, shown in the Index sheet's "Purpose"
# column. Keyed by the final (post-_safe_sheet_name) worksheet name, since
# that's what's actually written to the workbook.
_EXCEL_SHEET_PURPOSES: dict[str, str] = {
    "Config Summary": (
        "Every governance-relevant pipeline parameter — default vs. user choice, "
        "flagged where it deviates."
    ),
    "Model Summary": "Algorithm, task type, and training metadata for this run.",
    "Data Lineage": "Source data path, row/column counts, and train/val/test/OOT split sizes.",
    "EDA - Overview": "Dataset-level summary, flagged features, and recommended drops.",
    "EDA - Numeric": (
        "Descriptive statistics (mean, std, percentiles, skew, outlier rates) per numeric feature."
    ),
    "EDA - Categorical": (
        "Cardinality, mode, and frequency statistics for every categorical feature."
    ),
    "EDA - Datetime": "Date range and granularity statistics for every datetime feature.",
    "EDA - Boolean": "True/false distribution for every boolean feature.",
    "EDA - IV Ranking": "Information Value per feature, ranked, with a bar chart.",
    "EDA - Correlations": "Pearson correlation matrix with a conditional-format heatmap.",
    "EDA - Alerts": "Automatically flagged data-quality issues per feature.",
    "EDA - Missing": "Missing-value percentage by feature, with a bar chart.",
    "EDA - Missing Correlation": "Correlation between features' missingness patterns.",
    "EDA - Sample Rows": "Head and tail sample rows from the training set.",
    "EDA - Duplicate Rows": "Duplicate row groups found in the training set.",
    "EDA - Interactions": (
        "Pre-rendered scatter/hexbin images for the top IV-ranked numeric feature pairs "
        "(no raw data written to the sheet)."
    ),
    "EDA - Numeric Charts": (
        "Histogram (with KDE overlay where computable) for every numeric feature, "
        "extreme values clipped for readability."
    ),
    "EDA - Categorical Charts": (
        "Bar chart of the top value counts (with percentage) for every categorical feature, "
        "per split (Train/Test/Validation/OOT)."
    ),
    "Feature Inventory": (
        "Every feature considered for modelling, and whether it survived selection."
    ),
    "Performance Metrics": "Train/validation/test/OOT evaluation metrics.",
    "Decile Table": (
        "Decile-level event rate and cumulative lift per split "
        "(Train/Test/Validation/OOT), each with its own combo chart."
    ),
    "Stability": "Population Stability Index between training and out-of-time data.",
    "Calibration": "Calibration method and Expected Calibration Error before/after.",
    "Explainability": "SHAP/LIME feature importance.",
    "Hyperparameter Tuning": "Tuning configuration and best-trial results.",
    "Leaderboard": (
        "Comparison of every algorithm trained when leaderboard mode was enabled, "
        "ranked by the configured metric, plus per-algorithm metrics for every split "
        "(Train/Test/Validation/OOT)."
    ),
    "Limitations": "Known model limitations and caveats.",
    "Governance": "Approval status, owner, and review metadata.",
    "Full Config": "The complete YAML configuration used for this run.",
}


def _safe_sheet_name(name: str) -> str:
    """Sanitise a string into a valid Excel worksheet name.

    Args:
        name (str): Proposed worksheet name.

    Returns:
        str: ``name`` with every Excel-forbidden character (``: \\ / ? * [
        ]``) replaced by ``"_"``, truncated to
        ``_EXCEL_SHEET_NAME_MAX_LEN`` characters.
    """
    cleaned = "".join("_" if c in _EXCEL_FORBIDDEN_SHEET_CHARS else c for c in name)
    return cleaned[:_EXCEL_SHEET_NAME_MAX_LEN]


def _excel_col_letter(col_idx: int) -> str:
    """0-indexed column number to a single Excel column letter.

    Args:
        col_idx (int): 0-indexed column number. Every caller in this module
            stays within ``0``-``25`` (``_EXCEL_NAV_COL_OFFSET`` plus a
            handful of data columns), so multi-letter columns (``AA``, ...)
            are intentionally not handled.

    Returns:
        str: Single uppercase column letter, e.g. ``0 -> "A"``, ``1 -> "B"``.
    """
    return chr(ord("A") + col_idx)


class ModelCard:
    """Assembles and renders a governance-ready model card for any dscompanion
    model, covering model summary, data lineage, feature inventory,
    performance metrics, stability, calibration, explainability,
    hyperparameter tuning, limitations, and a sign-off governance table.

    All sections are assembled in-memory by calling ``generate()``, which
    populates the ``sections_`` attribute.  The card can then be serialised
    to HTML, Excel (via xlsxwriter), Word (.docx via python-docx), an
    artifact store, or a plain Python dict.

    Args:
        model: Fitted ``BaseDSCompanionModel`` instance whose ``estimator``,
            ``feature_names``, and ``evaluate`` attributes are introspected
            during card generation.
        split: ``DataSplit`` instance that provides ``X_train``, ``X_val``,
            ``X_oot``, and optionally ``metadata`` for the data lineage
            section. Anything that scores through the model (performance
            metrics, decile table, PSI, leaderboard per-split metrics)
            reads from this split, so in a ``PipelineRunner`` run it is the
            model-ready (post impute/encode/scale/select/imbalance) split.
        eda_split (optional): Pre-feature-engineering ``DataSplit`` — same
            shape as ``split``, but with original dtypes (categorical
            columns intact, unscaled numeric values). Used only by the
            per-split EDA sections (Numeric/Categorical/Missing/
            Categorical Charts/Interactions), which need human-readable
            raw values rather than the model's encoded/scaled input.
            Defaults to ``None``, which falls back to ``split`` itself —
            correct for the common case of constructing ``ModelCard``
            directly with one untransformed split (most tests, and any
            caller outside ``PipelineRunner``), but produces empty
            Test/Validation/OOT categorical sections when ``split`` is
            already fully numeric (post-encoding), since no categorical
            dtype columns remain to summarise.
        explainer (optional): Fitted ``SHAPExplainer`` instance.  When
            provided, the top-20 mean absolute SHAP importances are included
            in the explainability section.  Defaults to ``None``.
        permutation_importance (optional): Fitted
            ``PermutationImportanceAnalyser`` instance.  When provided, the
            top-20 permutation importances are included in the
            explainability section, independently of ``explainer`` — both
            may be provided together.  Defaults to ``None``.
        calibrator (optional): Fitted ``Calibrator`` instance.  When
            provided, the calibration report is included.  Defaults to
            ``None``.
        eda_report (optional): Fitted ``EDAReport`` instance (post
            ``run_all()``).  When provided, ``to_excel()`` writes a dedicated
            set of ``EDA - *`` worksheets from it.  Defaults to ``None`` (no
            EDA worksheets are written).
        tuner (optional): Fitted ``Tuner`` instance.  When provided, backend,
            number of trials, metric, best params, and best score are
            captured.  Defaults to ``None``.
        leaderboard (optional): ``Leaderboard`` instance after ``run()``.
            When provided, its ranked comparison table is included as the
            "Leaderboard" section (one row per algorithm compared).
            Defaults to ``None`` — the section still renders, as an empty
            (correctly-columned) table, matching the "sections are never
            skipped" convention used by ``decile_table``.
        author (str): Full name of the model author; appears in the summary
            and governance sign-off table.  Defaults to ``""``.
        model_version (str): Semantic version string for the model.
            Defaults to ``"1.0"``.
        use_case (str): Free-text description of the business use case.
            Defaults to ``""``.
        config_checks (list[dict[str, Any]] | None): Every monitored config
            parameter for this run — not just the ones that deviate — one
            dict per parameter with keys ``"parameter"``, ``"choices"``,
            ``"default"``, ``"user_choice"``, and ``"is_deviation"`` (bool).
            Typically ``PipelineRunner._build_full_config_table()``.
            Rendered as a table at the top of the report, before any
            metrics, so a reviewer can confirm "no deviations" by
            inspection rather than taking it on faith. Defaults to ``None``
            (section shows a note instead).
        decile_table (bool): Include a 10-decile gains/lift table, computed
            on ``split.X_test`` / ``split.y_test``. Classification only —
            silently omitted for regression/clustering models (no
            ``predict_proba``) or when this is ``False``. Defaults to
            ``True``, matching ``ReportingConfig.decile_table``.
        raw_config_yaml (str | None): The full experiment YAML as a string
            (typically ``PipelineConfig.to_yaml()``), rendered verbatim in a
            collapsible "Full Experiment Config" section at the bottom of
            the report — the exact YAML used, independent of which
            parameters ``config_checks`` chose to monitor. Defaults to
            ``None`` (section shows a note instead).
        excel_charts (bool): ``to_excel()``-only master switch for every
            native chart in the workbook (IV bar, correlation heatmap,
            missing-% bar, decile combo, interactions scatter, and the
            per-feature numeric/categorical distribution charts). Has no
            effect on ``to_word()``. Defaults to ``True``, matching
            ``EDAConfig.chart``.
        excel_chart_clip_lower_pct (float): ``to_excel()``-only lower-tail
            fraction clipped before building the per-feature numeric
            distribution histogram, so a few extreme values don't flatten
            the real distribution into one bin. Chart rendering only —
            never applied to modelling data. Defaults to ``0.01``, matching
            ``EDAConfig.chart_clip_lower_pct``.
        excel_chart_clip_upper_pct (float): ``to_excel()``-only upper-tail
            fraction clipped before building the per-feature numeric
            distribution histogram. Defaults to ``0.01``, matching
            ``EDAConfig.chart_clip_upper_pct``.

    Attributes:
        sections_ (Dict[str, Any]): Ordered dict of section key to section
            data (dict or ``pd.DataFrame``); populated after ``generate()``.
            Empty dict before ``generate()`` is called.
    """

    def __init__(
        self,
        model: Any,
        split: Any,
        eda_split: Any | None = None,
        explainer: Any | None = None,
        permutation_importance: Any | None = None,
        calibrator: Any | None = None,
        eda_report: Any | None = None,
        tuner: Any | None = None,
        leaderboard: Any | None = None,
        author: str = "",
        model_version: str = "1.0",
        use_case: str = "",
        config_checks: list[dict[str, Any]] | None = None,
        decile_table: bool = True,
        raw_config_yaml: str | None = None,
        excel_charts: bool = True,
        excel_chart_clip_lower_pct: float = 0.01,
        excel_chart_clip_upper_pct: float = 0.01,
    ) -> None:
        self.model = model
        self.split = split
        self.eda_split = eda_split if eda_split is not None else split
        self.explainer = explainer
        self.permutation_importance = permutation_importance
        self.calibrator = calibrator
        self.eda_report = eda_report
        self.tuner = tuner
        self.leaderboard = leaderboard
        self.author = author
        self.model_version = model_version
        self.use_case = use_case
        self.config_checks = config_checks
        self.decile_table = decile_table
        self.raw_config_yaml = raw_config_yaml
        self.excel_charts = excel_charts
        self.excel_chart_clip_lower_pct = excel_chart_clip_lower_pct
        self.excel_chart_clip_upper_pct = excel_chart_clip_upper_pct
        self.sections_: dict[str, Any] = {}

    def generate(self) -> "ModelCard":
        """Populate all model card sections by introspecting the model,
        split, explainer, calibrator, and tuner provided at construction,
        storing results in ``sections_``.

        Emits an INFO log line reporting the number of sections generated.
        Sections whose optional dependencies are absent (e.g. no calibrator)
        are filled with a ``{"note": "..."}`` placeholder dict rather than
        being omitted.

        Args:
            None

        Returns:
            ModelCard: The instance itself (``self``), enabling method
            chaining such as ``card.generate().to_excel("card.xlsx")``.
        """
        from dscompanion import __version__ as dscompanion_version

        self.sections_["config_summary"] = self._config_summary_section()

        self.sections_["model_summary"] = {
            "use_case": self.use_case,
            "task": type(self.model).__name__,
            "algorithm": type(self.model.estimator).__name__,
            "version": self.model_version,
            "author": self.author,
            "training_date": datetime.now().strftime("%Y-%m-%d"),
            "dscompanion_version": dscompanion_version,
            "python_version": sys.version.split()[0],
            "package_versions": self._key_package_versions(),
        }

        self.sections_["data_lineage"] = self._data_lineage()
        self.sections_["feature_inventory"] = self._feature_inventory()
        self.sections_["performance_metrics"] = self._performance_metrics()
        self.sections_["decile_table"] = self._decile_table_section()
        self.sections_["stability"] = self._stability()
        self.sections_["calibration"] = self._calibration_section()
        self.sections_["explainability"] = self._explainability()
        self.sections_["hyperparameter_tuning"] = self._tuning_section()
        self.sections_["leaderboard"] = self._leaderboard_section()
        self.sections_["limitations"] = {
            "notes": "",
            "boilerplate": (
                "This model was trained on historical data and may not generalise "
                "to future regimes. Performance should be monitored on a regular basis "
                "using PSI and KS metrics. The model has not been independently validated."
            ),
        }
        self.sections_["governance"] = pd.DataFrame(
            [
                {"role": "Author", "name": self.author, "date": "", "sign_off": ""},
                {"role": "Model Owner", "name": "", "date": "", "sign_off": ""},
                {"role": "Risk Review", "name": "", "date": "", "sign_off": ""},
                {"role": "Compliance", "name": "", "date": "", "sign_off": ""},
            ]
        )
        self.sections_["full_config"] = self._full_config_section()

        logger.info("ModelCard generated — %d sections", len(self.sections_))
        return self

    @staticmethod
    def _key_package_versions() -> dict[str, str]:
        """Return installed versions of the packages most relevant to
        reproducing or debugging this model card later.

        A version mismatch between the machine that generated a model card
        and wherever it's later debugged is a common source of confusion.
        Not an exhaustive environment dump — just
        ``_KEY_PACKAGES_FOR_VERSION_REPORT``, the training-library and
        report-generation stack.

        Args:
            None

        Returns:
            dict[str, str]: Package name -> installed version string. A
            package not installed in the current environment maps to
            ``"not installed"`` rather than being omitted, so the row set
            is stable across environments.
        """
        from importlib.metadata import PackageNotFoundError, version

        result: dict[str, str] = {}
        for package in _KEY_PACKAGES_FOR_VERSION_REPORT:
            try:
                result[package] = version(package)
            except PackageNotFoundError:
                result[package] = "not installed"
        return result

    def to_word(self, path: str | Path) -> Path:
        """Render the model card to a ``.docx`` Word document, writing each
        section as a heading with its data as paragraphs or a formatted table,
        and creating any missing parent directories automatically.

        Must be called after ``generate()``.

        Args:
            path (str or Path): Destination file path for the ``.docx``
                document.  Parent directories are created with
                ``mkdir(parents=True)`` if they do not already exist.

        Returns:
            pathlib.Path: The resolved ``Path`` of the written ``.docx``
            file.

        Raises:
            ImportError: If ``python-docx`` is not installed.
        """
        try:
            from docx import Document
        except ImportError as exc:
            raise ImportError("python-docx is required. pip install python-docx") from exc

        doc = Document()
        doc.add_heading("Model Card", 0)

        for section_key in _SECTION_KEYS:
            data = self.sections_.get(section_key)
            if data is None:
                continue
            doc.add_heading(section_key.replace("_", " ").title(), 1)
            if isinstance(data, dict):
                for k, v in data.items():
                    if not isinstance(v, (dict, pd.DataFrame)):
                        doc.add_paragraph(f"{k}: {v}")
            elif isinstance(data, pd.DataFrame):
                table = doc.add_table(rows=1, cols=len(data.columns))
                hdr = table.rows[0].cells
                for i, col in enumerate(data.columns):
                    hdr[i].text = str(col)
                for _, row in data.iterrows():
                    cells = table.add_row().cells
                    for i, val in enumerate(row):
                        cells[i].text = str(val)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(path)
        logger.info("ModelCard (docx) → %s", path)
        return path

    def to_html(self, path: str | Path) -> Path:
        """Render the model card to a self-contained HTML report.

        Must be called after ``generate()``. Format mirrors
        ``spark_data_profiler.py``: sticky Bootstrap 5 navbar, tabbed
        Overview, per-column variable cards (when ``eda_report`` was
        supplied), a sticky-header correlation heatmap, missing-values bar
        chart, sample rows — extended with the same visual language to
        every model-specific section (performance, decile/lift,
        calibration, explainability, stability, hyperparameter tuning,
        leaderboard, feature inventory, full config, governance). All
        charts are inline SVG; Bootstrap CSS/JS are fetched once per
        process and embedded inline for a single offline-viewable file
        (falls back to CDN tags if the fetch fails).

        Args:
            path (str or Path): Destination file path for the rendered
                HTML. Parent directories are created with
                ``mkdir(parents=True)`` if they do not already exist.

        Returns:
            pathlib.Path: The resolved ``Path`` of the written HTML file.
        """
        hw.ensure_bootstrap()

        has_eda = self.eda_report is not None
        eda_variables_html = eda_anchor_map = eda_alerts_html = ""
        total_alerts = 0
        eda_correlations_html = eda_missing_html = eda_sample_html = ""
        if has_eda:
            eda_variables_html, eda_anchor_map = html_eda.render_variable_cards(
                self.eda_report,
                self.excel_chart_clip_lower_pct,
                self.excel_chart_clip_upper_pct,
                settings.eda_histogram_bins,
            )
            eda_alerts_html, total_alerts = html_eda.render_alerts_tab(
                self.eda_report, eda_anchor_map
            )
            eda_correlations_html = html_eda.render_correlations_section(
                self.eda_report, eda_anchor_map
            )
            eda_missing_html = html_eda.render_missing_section(self.eda_report, eda_anchor_map)
            eda_sample_html = html_eda.render_sample_section(self.eda_report)

        overview_html = html_model.render_overview_section(
            self.sections_.get("model_summary", {}),
            self.sections_.get("data_lineage", {}),
            self.sections_.get("config_summary", pd.DataFrame()),
        )
        overview_alerts_tab = eda_alerts_html or (
            '<p class="text-muted p-3">No EDA report was provided for this run.</p>'
        )
        overview_schema_tab = (
            html_eda.render_schema_tab(
                self.split, list(self.sections_["feature_inventory"]["feature"])
            )
            if not self.sections_["feature_inventory"].empty
            else ""
        )
        overview_eda_tab = (
            html_eda.render_overview_tab(self.eda_report)
            if has_eda
            else '<p class="text-muted p-3">No EDA report was provided for this run.</p>'
        )

        performance_html = html_model.render_performance_section(
            self.sections_["performance_metrics"]
        )
        decile_html = html_model.render_decile_section(self.sections_["decile_table"])
        calibration_html = html_model.render_calibration_section(self.sections_["calibration"])
        explainability_html = html_model.render_explainability_section(
            self.sections_["explainability"]
        )
        stability_html = html_model.render_stability_section(self.sections_["stability"])
        tuning_html = html_model.render_tuning_section(self.sections_["hyperparameter_tuning"])
        leaderboard_html = html_model.render_leaderboard_section(self.sections_["leaderboard"])
        feature_inventory_html = html_model.render_feature_inventory_section(
            self.sections_["feature_inventory"]
        )
        config_summary_html = html_model.render_config_summary_section(
            self.sections_["config_summary"]
        )
        full_config_html = html_model.render_full_config_section(self.sections_["full_config"])
        governance_html = html_model.render_governance_section(
            self.sections_["governance"], self.sections_["limitations"]
        )

        title = _html_lib.escape(f"Model Card — {self.use_case or self.model_version}")

        nav_links = [("#overview", "Overview")]
        if has_eda:
            nav_links.append(("#eda-variables", "Variables"))
            if eda_correlations_html:
                nav_links.append(("#eda-correlations", "Correlations"))
            nav_links.append(("#eda-missing", "Missing values"))
            if eda_sample_html:
                nav_links.append(("#eda-sample", "Sample"))
        nav_links += [
            ("#performance", "Performance"),
            ("#decile", "Decile / Lift"),
            ("#calibration", "Calibration"),
            ("#explainability", "Explainability"),
            ("#stability", "Stability"),
            ("#tuning", "Tuning"),
            ("#leaderboard", "Leaderboard"),
            ("#feature-inventory", "Feature Inventory"),
            ("#full-config", "Full Config"),
            ("#governance", "Governance"),
        ]
        nav_html = "".join(
            f'<li class="nav-item"><a class="nav-link" href="{href}">{label}</a></li>'
            for href, label in nav_links
        )

        eda_section_html = ""
        if has_eda:
            eda_section_html = f"""
  <div class="section-header" id="eda-variables"><h1 class="section-name">Variables</h1></div>
  <div class="section-items">{eda_variables_html}</div>
  {'<div class="section-header" id="eda-correlations">'
   '<h1 class="section-name">Correlations</h1></div>'
   '<div class="section-items"><div class="row item">' + eda_correlations_html + '</div></div>'
   if eda_correlations_html else ''}
  <div class="section-header" id="eda-missing"><h1 class="section-name">Missing values</h1></div>
  <div class="section-items"><div class="row item">{eda_missing_html}</div></div>
  {'<div class="section-header" id="eda-sample"><h1 class="section-name">Sample</h1></div>'
   '<div class="section-items"><div class="row item">' + eda_sample_html + '</div></div>'
   if eda_sample_html else ''}"""

        body = f"""
<nav class="navbar navbar-expand-lg bg-body-tertiary sticky-top">
  <div class="container-fluid">
    <a class="navbar-brand" href="#overview">{title}</a>
    <button class="navbar-toggler" type="button" data-bs-toggle="collapse"
            data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false"
            aria-label="Toggle navigation"><span class="navbar-toggler-icon"></span></button>
    <div class="collapse navbar-collapse" id="navbarNav">
      <ul class="navbar-nav ms-auto gap-2">{nav_html}</ul>
    </div>
  </div>
</nav>
<div class="container mt-3">
  <div class="section-header" id="overview"><h1 class="section-name">Overview</h1></div>
  <div class="section-items">
    <div class="row item">
      <ul class="nav nav-tabs" role="tablist">
        <li class="nav-item"><button class="nav-link active" data-bs-toggle="tab"
            data-bs-target="#pane-ov-overview" type="button">Overview</button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="tab"
            data-bs-target="#pane-ov-eda" type="button">EDA Summary</button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="tab"
            data-bs-target="#pane-ov-alerts" type="button">Alerts
            <span class="badge text-bg-secondary">{total_alerts}</span></button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="tab"
            data-bs-target="#pane-ov-schema" type="button">Schema</button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="tab"
            data-bs-target="#pane-ov-config" type="button">Config Summary</button></li>
      </ul>
      <div class="tab-content">
        <div class="tab-pane fade show active" id="pane-ov-overview">{overview_html}</div>
        <div class="tab-pane fade" id="pane-ov-eda">{overview_eda_tab}</div>
        <div class="tab-pane fade" id="pane-ov-alerts">{overview_alerts_tab}</div>
        <div class="tab-pane fade" id="pane-ov-schema">{overview_schema_tab}</div>
        <div class="tab-pane fade" id="pane-ov-config">{config_summary_html}</div>
      </div>
    </div>
  </div>
  {eda_section_html}
  <div class="section-header" id="performance"><h1 class="section-name">Performance</h1></div>
  <div class="section-items"><div class="row item">{performance_html}</div></div>
  <div class="section-header" id="decile"><h1 class="section-name">Decile / Lift</h1></div>
  <div class="section-items"><div class="row item">{decile_html}</div></div>
  <div class="section-header" id="calibration"><h1 class="section-name">Calibration</h1></div>
  <div class="section-items"><div class="row item">{calibration_html}</div></div>
  <div class="section-header" id="explainability"><h1 class="section-name">Explainability</h1></div>
  <div class="section-items"><div class="row item">{explainability_html}</div></div>
  <div class="section-header" id="stability"><h1 class="section-name">Stability</h1></div>
  <div class="section-items"><div class="row item">{stability_html}</div></div>
  <div class="section-header" id="tuning"><h1 class="section-name">Hyperparameter Tuning</h1></div>
  <div class="section-items"><div class="row item">{tuning_html}</div></div>
  <div class="section-header" id="leaderboard"><h1 class="section-name">Leaderboard</h1></div>
  <div class="section-items"><div class="row item">{leaderboard_html}</div></div>
  <div class="section-header" id="feature-inventory">
    <h1 class="section-name">Feature Inventory</h1></div>
  <div class="section-items"><div class="row item">{feature_inventory_html}</div></div>
  <div class="section-header" id="full-config"><h1 class="section-name">Full Config</h1></div>
  <div class="section-items"><div class="row item">{full_config_html}</div></div>
  <div class="section-header" id="governance"><h1 class="section-name">Governance</h1></div>
  <div class="section-items"><div class="row item">{governance_html}</div></div>
  <footer class="mt-4">
    <p class="text-body-secondary text-center small">Report generated by
      <strong>dscompanion ModelCard</strong></p>
  </footer>
</div>"""

        html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,shrink-to-fit=no">
<title>{title}</title>
{hw.bootstrap_css()}
<style>
:root {{ --navbar-height: 56px; }}
html {{ scroll-padding-top: var(--navbar-height); }}
body {{ padding-bottom: 3rem; }}
.section-header {{ margin-top: 1rem; margin-bottom: 1rem; }}
.section-name {{ font-size: 1.5rem; font-weight: 500; }}
.section-items > .row {{
  padding: 1rem; margin: 0 0 1rem;
  border: 1px solid var(--bs-border-color);
  border-radius: var(--bs-border-radius); }}
.item-header {{ margin-top: .5rem; margin-bottom: .5rem; }}
.row.sub-item {{ margin-left: 0 !important; margin-right: 0 !important; }}
.row.sub-item > * {{ padding-left: 0 !important; padding-right: 0 !important; }}
div[class*="col-"] + div[class*="col-"] {{ padding-left: 1rem !important; }}
.alert-info > td {{ color: var(--bs-danger); }}
</style>
</head>
<body>
{body}
{hw.bootstrap_js()}
</body>
</html>"""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        logger.info("ModelCard (html) → %s", path)
        return path

    def to_excel(self, path: str | Path) -> Path:
        """Render the model card to a multi-sheet Excel workbook via xlsxwriter.

        Must be called after ``generate()``. Writes one worksheet per
        top-level section (the 13 keys of ``_SECTION_KEYS``), plus — when an
        ``EDAReport`` was supplied — a dedicated set of ``EDA - *``
        sub-sheets built directly from it (Overview, Numeric, Categorical,
        Datetime, Boolean, IV Ranking, Correlations, Alerts, Missing,
        Missing Correlation, Sample Rows, Duplicate Rows, Interactions).
        Charts and conditional-format heatmaps are native xlsxwriter
        constructs built from already-fetched DataFrames, avoiding a
        ``kaleido`` dependency for Plotly-to-image conversion.

        Args:
            path (str or Path): Destination ``.xlsx`` path. Parent
                directories are created with ``mkdir(parents=True)`` if
                they do not already exist.

        Returns:
            pathlib.Path: The resolved ``Path`` of the written workbook.

        Raises:
            ImportError: If ``xlsxwriter`` is not installed.
        """
        try:
            import xlsxwriter  # noqa: F401
        except ImportError as exc:
            raise ImportError("xlsxwriter is required. pip install xlsxwriter") from exc

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
            # Created first so it lands as the first tab; populated last,
            # once every other sheet has been registered.
            index_worksheet = writer.book.add_worksheet(_EXCEL_INDEX_SHEET_NAME)
            index_worksheet.hide_gridlines(2)
            self._excel_sheet_registry: list[str] = []

            for section_key in _SECTION_KEYS:
                if section_key == "feature_inventory":
                    self._write_eda_excel_sheets(writer)
                data = self.sections_.get(section_key)
                if data is None:
                    continue
                sheet_name = _safe_sheet_name(section_key.replace("_", " ").title())
                if section_key == "performance_metrics":
                    self._write_performance_metrics_excel_sheet(writer, sheet_name)
                elif section_key == "decile_table":
                    self._write_decile_table_excel_sheet(writer, sheet_name)
                elif section_key == "leaderboard":
                    self._write_leaderboard_excel_sheet(writer, sheet_name)
                else:
                    self._write_section_excel_sheet(writer, sheet_name, data)
                self._register_excel_sheet(writer, sheet_name)

            self._write_excel_index(index_worksheet)

        logger.info("ModelCard (xlsx) → %s", path)
        return path

    def _register_excel_sheet(self, writer: Any, sheet_name: str) -> None:
        """Hide gridlines and write the "Back to Index" nav link at A1.

        Call exactly once per worksheet, immediately after its first write,
        so ``writer.sheets[sheet_name]`` already exists. Records
        ``sheet_name`` in ``self._excel_sheet_registry`` for later use by
        ``_write_excel_index``.

        Args:
            writer (pd.ExcelWriter): Open writer using the ``xlsxwriter``
                engine.
            sheet_name (str): Name of the worksheet that was just written.

        Returns:
            None
        """
        worksheet = writer.sheets[sheet_name]
        worksheet.hide_gridlines(2)
        worksheet.write_url(
            0,
            _EXCEL_NAV_COL_OFFSET,
            f"internal:'{_EXCEL_INDEX_SHEET_NAME}'!A1",
            string=_EXCEL_NAV_LINK_TEXT,
        )
        self._excel_sheet_registry.append(sheet_name)

    def _write_excel_index(self, worksheet: Any) -> None:
        """Populate the ``Index`` sheet's navigation table.

        Writes one row per worksheet registered via ``_register_excel_sheet``
        (in the order they were written), each with a hyperlink to that
        sheet's ``A1`` cell, a one-line purpose description, and a blank
        ``Feedback`` cell left for reviewer comments.

        Args:
            worksheet (Any): The ``xlsxwriter.Worksheet`` for ``Index``,
                created at the start of ``to_excel()``.

        Returns:
            None
        """
        header_row = _EXCEL_NAV_ROW_OFFSET
        col = _EXCEL_NAV_COL_OFFSET
        worksheet.write_row(header_row, col, ["Sheet", "Purpose", "Feedback"])
        for offset, sheet_name in enumerate(self._excel_sheet_registry, start=1):
            row = header_row + offset
            worksheet.write_url(row, col, f"internal:'{sheet_name}'!A1", string=sheet_name)
            purpose = _EXCEL_SHEET_PURPOSES.get(sheet_name, "")
            worksheet.write(row, col + 1, purpose)
        worksheet.set_column(col, col, 28)
        worksheet.set_column(col + 1, col + 1, 80)
        worksheet.set_column(col + 2, col + 2, 40)

    def _write_section_excel_sheet(self, writer: Any, sheet_name: str, data: Any) -> None:
        """Write one non-EDA model-card section to a single Excel worksheet.

        Dispatches on ``data``'s shape: a ``pd.DataFrame`` is written
        directly; a ``dict`` has its scalar values written as a
        ``parameter, value`` table, followed by any nested
        ``dict``/``pd.DataFrame``/``list`` values as additional stacked
        tables on the same sheet (a lone ``{"yaml": ...}`` value is
        written as a single wrapped multi-line cell instead of exploded
        into rows). A dict with only a ``{"note": ...}`` key still gets a
        one-row sheet — sections are never skipped, matching ``to_word()``.
        A nested ``dict`` value (e.g. ``hyperparameter_tuning``'s
        ``best_params``) is itself rendered as a ``parameter, value``
        table, not passed straight to ``pd.DataFrame()`` — a flat dict of
        scalars has no array-like values for pandas to build columns
        from, which raises ``ValueError: If using all scalar values, you
        must pass an index``.

        Args:
            writer (pd.ExcelWriter): Open writer using the ``xlsxwriter``
                engine.
            sheet_name (str): Already-sanitised worksheet name (≤31 chars,
                no Excel-forbidden characters).
            data (pd.DataFrame or dict): Section payload from
                ``self.sections_``.

        Returns:
            None
        """
        col = _EXCEL_NAV_COL_OFFSET
        if isinstance(data, pd.DataFrame):
            data.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
                startrow=_EXCEL_NAV_ROW_OFFSET,
                startcol=col,
            )
            return

        scalar_items = {
            k: v for k, v in data.items() if not isinstance(v, (dict, pd.DataFrame, list))
        }
        nested_items = {k: v for k, v in data.items() if isinstance(v, (dict, pd.DataFrame, list))}

        scalar_df = pd.DataFrame(list(scalar_items.items()), columns=["parameter", "value"])
        scalar_df.to_excel(
            writer, sheet_name=sheet_name, index=False, startrow=_EXCEL_NAV_ROW_OFFSET, startcol=col
        )
        start_row = _EXCEL_NAV_ROW_OFFSET + len(scalar_df) + 2

        for key, value in nested_items.items():
            if key == "yaml":
                worksheet = writer.sheets[sheet_name]
                wrap_format = writer.book.add_format({"text_wrap": True})
                worksheet.write_string(start_row, col, str(value), wrap_format)
                start_row += 2
                continue
            if isinstance(value, pd.DataFrame):
                nested_df = value
            elif isinstance(value, dict):
                nested_df = pd.DataFrame(list(value.items()), columns=["parameter", "value"])
            else:
                nested_df = pd.DataFrame(value)
            nested_df.to_excel(
                writer, sheet_name=sheet_name, index=False, startrow=start_row, startcol=col
            )
            start_row += len(nested_df) + 2

    def _write_performance_metrics_excel_sheet(self, writer: Any, sheet_name: str) -> None:
        """Write the Performance Metrics sheet: comparison table plus diagnostics.

        Bespoke writer (mirrors the EDA sheets' own dedicated-writer
        precedent) rather than an extension of the generic
        ``_write_section_excel_sheet`` — that generic handler has no way to
        color-code diagnostic flags by severity. Falls back to the generic
        writer (with a "no metrics" note) when ``_comparison_metrics_table()``
        is empty, so the sheet is never silently skipped.

        Args:
            writer (pd.ExcelWriter): Open writer using the ``xlsxwriter``
                engine.
            sheet_name (str): Already-sanitised worksheet name.

        Returns:
            None
        """
        comparison = self._comparison_metrics_table()
        if comparison.empty:
            self._write_section_excel_sheet(
                writer, sheet_name, self.sections_.get("performance_metrics", pd.DataFrame())
            )
            return

        bold = writer.book.add_format({"bold": True})
        row = _EXCEL_NAV_ROW_OFFSET
        col = _EXCEL_NAV_COL_OFFSET

        comparison.to_excel(
            writer, sheet_name=sheet_name, index=False, startrow=row + 1, startcol=col
        )
        worksheet = writer.sheets[sheet_name]
        worksheet.write(row, col, "Comparison (all splits)", bold)
        row += len(comparison) + 3

        flags = self._performance_diagnostics()
        flag_format = {
            "error": writer.book.add_format({"bold": True, "font_color": "#C00000"}),
            "warning": writer.book.add_format({"bold": True, "font_color": "#9C6500"}),
            "info": writer.book.add_format({"font_color": "#1F4E78"}),
        }
        worksheet.write(row, col, "Diagnostics", bold)
        row += 1
        if flags:
            for flag in flags:
                worksheet.write(row, col, f"[{flag['level'].upper()}]", flag_format[flag["level"]])
                worksheet.write(row, col + 1, flag["message"])
                row += 1
        else:
            worksheet.write(row, col, "No suspicious train/test/validation/OOT patterns detected.")
            row += 1

    def _write_decile_table_excel_sheet(self, writer: Any, sheet_name: str) -> None:
        """Write the Decile Table sheet as one labelled, charted section per split.

        Mirrors ``_write_performance_metrics_excel_sheet()``'s bespoke-writer
        shape, extended with one ``_add_decile_chart()`` per split (the
        original single-table sheet's chart, now repeated per split instead
        of test-only). Extra row spacing (vs. the plain metrics sheet)
        leaves room for each chart's default height so consecutive charts
        don't visually overlap. Falls back to the generic writer when
        there's nothing to show.

        Args:
            writer (pd.ExcelWriter): Open writer using the ``xlsxwriter``
                engine.
            sheet_name (str): Already-sanitised worksheet name.

        Returns:
            None
        """
        by_split = self._decile_table_by_split()
        if not by_split:
            self._write_section_excel_sheet(
                writer, sheet_name, self.sections_.get("decile_table", pd.DataFrame())
            )
            return

        bold = writer.book.add_format({"bold": True})
        row = _EXCEL_NAV_ROW_OFFSET
        col = _EXCEL_NAV_COL_OFFSET
        for label, split_df in by_split.items():
            table_start_row = row + 1
            split_df.to_excel(
                writer, sheet_name=sheet_name, index=False, startrow=table_start_row, startcol=col
            )
            writer.sheets[sheet_name].write(row, col, label, bold)
            if self.excel_charts:
                self._add_decile_chart(
                    writer.book,
                    writer.sheets[sheet_name],
                    sheet_name,
                    len(split_df),
                    start_row=table_start_row,
                    start_col=col,
                )
            # A default xlsxwriter chart is ~15-16 rows tall; a bare decile
            # table (10 rows) needs extra headroom so the next section's
            # chart doesn't overlap this one's.
            row += max(len(split_df) + 3, 18)

    def _write_leaderboard_excel_sheet(self, writer: Any, sheet_name: str) -> None:
        """Write the Leaderboard sheet: the ranking table, then a
        per-algorithm, per-split metrics table appended below.

        Excel-only: ``Leaderboard.leaderboard_`` and its ranking (based on
        a single resolved ``eval_split``) are untouched — this section
        reuses ``Leaderboard.fitted_models_`` (already fitted; no
        refitting) to show every algorithm's metrics across every split,
        so a train/test/OOT gap is visible per algorithm, not only on the
        ranking split.

        Args:
            writer (pd.ExcelWriter): Open writer using the ``xlsxwriter``
                engine.
            sheet_name (str): Already-sanitised worksheet name.

        Returns:
            None
        """
        data = self.sections_.get("leaderboard", pd.DataFrame())
        self._write_section_excel_sheet(writer, sheet_name, data)

        by_split = self._leaderboard_metrics_by_split()
        if not by_split:
            return

        worksheet = writer.sheets[sheet_name]
        bold = writer.book.add_format({"bold": True})
        col = _EXCEL_NAV_COL_OFFSET
        row = _EXCEL_NAV_ROW_OFFSET + len(data) + 3
        worksheet.write(row, col, "Per-Algorithm Metrics — All Splits", bold)
        row += 1
        for label, split_df in by_split.items():
            split_df.to_excel(
                writer, sheet_name=sheet_name, index=False, startrow=row + 1, startcol=col
            )
            worksheet.write(row, col, label, bold)
            row += len(split_df) + 3

    def _write_eda_excel_sheets(self, writer: Any) -> None:
        """Write one Excel worksheet per EDA Summary sub-table, plus native charts.

        Reads directly from ``self.eda_report``'s accessor methods.
        Sheets for empty/not-applicable sub-tables (no datetime columns, no
        alerts, compliance gates left at their default ``False``, etc.) are
        skipped entirely — never written empty.

        Args:
            writer (pd.ExcelWriter): Open writer using the ``xlsxwriter``
                engine.

        Returns:
            None
        """
        eda = self.eda_report
        if eda is None:
            return
        workbook = writer.book

        offset = _EXCEL_NAV_ROW_OFFSET
        col = _EXCEL_NAV_COL_OFFSET

        overview_items = list(eda.overview_summary().items())
        overview_df = pd.DataFrame(overview_items, columns=["metric", "value"])
        overview_df.to_excel(
            writer, sheet_name="EDA - Overview", index=False, startrow=offset, startcol=col
        )
        self._register_excel_sheet(writer, "EDA - Overview")
        start_row = offset + len(overview_df) + 2
        flagged_rows = [
            {"category": category, "feature": feature}
            for category, features in eda.flagged_features().items()
            for feature in features
        ]
        pd.DataFrame(flagged_rows, columns=["category", "feature"]).to_excel(
            writer, sheet_name="EDA - Overview", index=False, startrow=start_row, startcol=col
        )
        start_row += len(flagged_rows) + 2
        pd.DataFrame({"recommended_drops": eda.recommended_drops}).to_excel(
            writer, sheet_name="EDA - Overview", index=False, startrow=start_row, startcol=col
        )

        numeric_df = eda.numeric_summary()
        categorical_df = eda.categorical_summary()
        numeric_by_split = self._eda_summary_by_split(numeric_df, "numeric")
        categorical_by_split = self._eda_summary_by_split(categorical_df, "categorical")
        if numeric_by_split:
            self._write_stacked_eda_sections(writer, "EDA - Numeric", numeric_by_split)
        if categorical_by_split:
            self._write_stacked_eda_sections(writer, "EDA - Categorical", categorical_by_split)

        datetime_df = eda.datetime_summary()
        if not datetime_df.empty:
            datetime_df.to_excel(
                writer, sheet_name="EDA - Datetime", index=False, startrow=offset, startcol=col
            )
            self._register_excel_sheet(writer, "EDA - Datetime")

        boolean_df = eda.boolean_summary()
        if not boolean_df.empty:
            boolean_df.to_excel(
                writer, sheet_name="EDA - Boolean", index=False, startrow=offset, startcol=col
            )
            self._register_excel_sheet(writer, "EDA - Boolean")

        iv_df = eda.iv_table()
        if not iv_df.empty:
            sheet_name = "EDA - IV Ranking"
            iv_df.to_excel(
                writer, sheet_name=sheet_name, index=False, startrow=offset, startcol=col
            )
            self._register_excel_sheet(writer, sheet_name)
            if self.excel_charts:
                self._add_bar_chart(
                    workbook,
                    writer.sheets[sheet_name],
                    sheet_name,
                    n_rows=len(iv_df),
                    category_col=col,
                    value_col=col + 1,
                    title="Information Value by Feature",
                    start_row=offset,
                )

        corr_df = eda.correlation_table()
        if not corr_df.empty:
            sheet_name = "EDA - Correlations"
            corr_to_write = corr_df.reset_index().rename(columns={"index": "feature"})
            corr_to_write.to_excel(
                writer, sheet_name=sheet_name, index=False, startrow=offset, startcol=col
            )
            self._register_excel_sheet(writer, sheet_name)
            if self.excel_charts:
                n_rows, n_cols = corr_df.shape
                writer.sheets[sheet_name].conditional_format(
                    offset + 1,
                    col + 1,
                    offset + n_rows,
                    col + n_cols,
                    {
                        "type": "3_color_scale",
                        "min_color": _EXCEL_HEATMAP_COLORS["min"],
                        "mid_color": _EXCEL_HEATMAP_COLORS["mid"],
                        "max_color": _EXCEL_HEATMAP_COLORS["max"],
                    },
                )

        alert_rows = [
            {"feature": feature, "type": alert["type"], "detail": alert["detail"]}
            for feature, alert_list in eda.alerts().items()
            for alert in alert_list
        ]
        if alert_rows:
            pd.DataFrame(alert_rows, columns=["feature", "type", "detail"]).to_excel(
                writer, sheet_name="EDA - Alerts", index=False, startrow=offset, startcol=col
            )
            self._register_excel_sheet(writer, "EDA - Alerts")

        self._write_eda_missing_excel_sheet(writer, numeric_by_split, categorical_by_split)

        missing_corr_df = eda.missing_correlation()
        if not missing_corr_df.empty:
            sheet_name = "EDA - Missing Correlation"
            missing_corr_df.reset_index().rename(columns={"index": "feature"}).to_excel(
                writer, sheet_name=sheet_name, index=False, startrow=offset, startcol=col
            )
            self._register_excel_sheet(writer, sheet_name)

        sample = eda.sample_rows()
        if not sample["head"].empty:
            sheet_name = "EDA - Sample Rows"
            sample["head"].to_excel(
                writer, sheet_name=sheet_name, index=False, startrow=offset, startcol=col
            )
            self._register_excel_sheet(writer, sheet_name)
            sample["tail"].to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
                startrow=offset + len(sample["head"]) + 2,
                startcol=col,
            )

        duplicate_df = eda.duplicate_row_content()
        if not duplicate_df.empty:
            sheet_name = "EDA - Duplicate Rows"
            duplicate_df.to_excel(
                writer, sheet_name=sheet_name, index=False, startrow=offset, startcol=col
            )
            self._register_excel_sheet(writer, sheet_name)

        if self.excel_charts:
            self._write_interactions_excel_sheet(writer)
            self._write_numeric_charts_excel_sheet(writer, numeric_df)
            self._write_categorical_charts_excel_sheet(writer, categorical_df)

    def _eda_summary_by_split(self, train_df: pd.DataFrame, kind: str) -> dict[str, pd.DataFrame]:
        """Build ``{label: summary_df}`` for every split with data, Train first.

        ``Train`` reuses the already-computed ``train_df`` (``self.eda_report``'s
        own ``numeric_summary()``/``categorical_summary()``, at EDA time,
        whatever thresholds it was configured with); Test/Validation/OOT
        are freshly computed via a plain ``UnivariateAnalyser`` (global
        ``settings.*`` thresholds — ``EDAReport``'s own threshold
        overrides, if any, are private and not recoverable here) so a
        large stat drift between splits is visible on one sheet instead of
        only ever seeing Train.

        Reads ``self.eda_split`` — not ``self.split`` — for Test/Validation/
        OOT: in a ``PipelineRunner`` run, ``self.split`` is the model-ready,
        already-encoded split (no categorical dtype columns remain), which
        would silently produce empty Test/Validation/OOT categorical
        sections and, for Numeric, would compare genuinely-raw Train
        columns against Test/Validation/OOT columns that were mostly
        formerly-categorical encoded values.

        Args:
            train_df (pd.DataFrame): Train's already-computed summary
                (``numeric_summary()`` or ``categorical_summary()`` output).
            kind (str): ``"numeric"`` or ``"categorical"`` — selects which
                ``UnivariateAnalyser`` accessor to call for the other splits.

        Returns:
            dict[str, pd.DataFrame]: Keyed by ``"Train"``, ``"Test"``,
            ``"Validation"``, ``"OOT"`` (only splits with data are
            included, in that order). Empty dict if ``train_df`` is empty.
        """
        if train_df.empty:
            return {}
        from types import SimpleNamespace

        from dscompanion.eda import UnivariateAnalyser

        result: dict[str, pd.DataFrame] = {"Train": train_df}
        other_specs = [
            ("Test", self.eda_split.X_test),
            ("Validation", getattr(self.eda_split, "X_val", None)),
            ("OOT", getattr(self.eda_split, "X_oot", None)),
        ]
        for label, X in other_specs:
            if X is None or len(X) == 0:
                continue
            try:
                analyser = UnivariateAnalyser().fit(SimpleNamespace(train_X=X))
                split_df = (
                    analyser.numeric_summary()
                    if kind == "numeric"
                    else analyser.categorical_summary()
                )
            except Exception as exc:
                logger.debug("EDA %s summary (%s) skipped: %s", kind, label, exc)
                continue
            if not split_df.empty:
                result[label] = split_df
        return result

    def _write_stacked_eda_sections(
        self, writer: Any, sheet_name: str, by_split: dict[str, pd.DataFrame]
    ) -> None:
        """Write one bold-labelled section per split, tables stacked top to bottom.

        Shared by the per-split EDA sheets that need no chart (Numeric,
        Categorical) — same labelled-stacking convention as the Decile
        Table / Performance Metrics sheets, minus the chart-insertion step.

        Args:
            writer (pd.ExcelWriter): Open writer using the ``xlsxwriter``
                engine.
            sheet_name (str): Already-sanitised worksheet name.
            by_split (dict[str, pd.DataFrame]): Output of
                ``_eda_summary_by_split()``. Must be non-empty — callers
                check this before calling.

        Returns:
            None
        """
        bold = writer.book.add_format({"bold": True})
        row = _EXCEL_NAV_ROW_OFFSET
        col = _EXCEL_NAV_COL_OFFSET
        registered = False
        for label, split_df in by_split.items():
            split_df.to_excel(
                writer, sheet_name=sheet_name, index=False, startrow=row + 1, startcol=col
            )
            if not registered:
                self._register_excel_sheet(writer, sheet_name)
                registered = True
            writer.sheets[sheet_name].write(row, col, label, bold)
            row += len(split_df) + 3

    def _write_eda_missing_excel_sheet(
        self,
        writer: Any,
        numeric_by_split: dict[str, pd.DataFrame],
        categorical_by_split: dict[str, pd.DataFrame],
    ) -> None:
        """Write the ``EDA - Missing`` sheet as one labelled, charted section per split.

        R.6: extends the original train-only missing-% table/chart to
        Train/Test/Validation/OOT. Missing % is derived from the same
        per-split numeric/categorical summaries used by the ``EDA -
        Numeric``/``EDA - Categorical`` sheets, so all three sheets agree
        about what "Test"/"Validation"/"OOT" measures.

        Args:
            writer (pd.ExcelWriter): Open writer using the ``xlsxwriter``
                engine.
            numeric_by_split (dict[str, pd.DataFrame]): Output of
                ``_eda_summary_by_split(numeric_df, "numeric")``.
            categorical_by_split (dict[str, pd.DataFrame]): Output of
                ``_eda_summary_by_split(categorical_df, "categorical")``.

        Returns:
            None
        """
        missing_cols = ["feature", "missing_pct"]
        labels = list(dict.fromkeys([*numeric_by_split, *categorical_by_split]))
        by_split: dict[str, pd.DataFrame] = {}
        for label in labels:
            parts = [
                df[missing_cols]
                for df in (numeric_by_split.get(label), categorical_by_split.get(label))
                if df is not None
            ]
            if not parts:
                continue
            combined = pd.concat(parts, ignore_index=True).sort_values(
                "missing_pct", ascending=False
            )
            if not combined.empty:
                by_split[label] = combined
        if not by_split:
            return

        sheet_name = "EDA - Missing"
        bold = writer.book.add_format({"bold": True})
        row = _EXCEL_NAV_ROW_OFFSET
        col = _EXCEL_NAV_COL_OFFSET
        registered = False
        for label, split_df in by_split.items():
            table_start_row = row + 1
            split_df.to_excel(
                writer, sheet_name=sheet_name, index=False, startrow=table_start_row, startcol=col
            )
            if not registered:
                self._register_excel_sheet(writer, sheet_name)
                registered = True
            worksheet = writer.sheets[sheet_name]
            worksheet.write(row, col, label, bold)
            if self.excel_charts:
                self._add_bar_chart(
                    writer.book,
                    worksheet,
                    sheet_name,
                    n_rows=len(split_df),
                    category_col=col,
                    value_col=col + 1,
                    title=f"Missing % by Feature — {label}",
                    start_row=table_start_row,
                )
            # A default xlsxwriter chart is ~15-16 rows tall; leave headroom
            # so the next section's chart doesn't visually overlap this one's.
            row += max(len(split_df) + 3, 18)

    def _write_interactions_excel_sheet(self, writer: Any) -> None:
        """Write the ``EDA - Interactions`` sheet: one pre-rendered scatter/hexbin
        image per numeric feature pair.

        Renders each pair as a ``matplotlib``/``seaborn`` PNG, embedded via
        ``worksheet.insert_image()`` — no raw data written to any cell, so
        the sheet's size doesn't scale with row count or combinatorially
        with column count (``C(k, 2)`` pairs), unlike writing the raw
        feature slice into cells for a native xlsxwriter chart to
        reference. Mirrors ``BivariateAnalyser.interaction_plots()``'s own
        scatter/hexbin threshold switch
        (``settings.eda_interaction_hexbin_row_threshold``), so a very
        dense pair renders as a readable density plot instead of an
        overplotted scatter; that HTML-path implementation can't be reused
        directly here since it returns Plotly figures, and rendering those
        to a static image needs an additional dependency (``kaleido``) that
        this module otherwise has no reason to require.

        Uses the same IV-ranked/capped column selection as the HTML
        ``interaction_plots()`` (via ``EDAReport.interaction_columns()``),
        reading raw values from ``self.eda_split.X_train`` — not
        ``self.split``, which in a ``PipelineRunner`` run is already
        imputed/encoded/scaled and would plot transformed values instead of
        the original ones. ``interaction_columns()`` reflects the EDA-time
        column set, which can include columns dropped later by feature
        selection — these are filtered out before subsetting. Row count is
        further capped at ``settings.max_eda_rows`` before plotting,
        matching ``UnivariateAnalyser``'s own sampling behaviour.

        Args:
            writer (pd.ExcelWriter): Open writer using the ``xlsxwriter``
                engine.

        Returns:
            None
        """
        train_cols = self.eda_split.X_train.columns
        columns = [c for c in self.eda_report.interaction_columns() if c in train_cols]
        if len(columns) < 2:
            return

        import io

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        data = self.eda_split.X_train[columns]
        if len(data) > settings.max_eda_rows:
            data = data.sample(n=settings.max_eda_rows, random_state=settings.random_state)
        use_hexbin = len(data) > settings.eda_interaction_hexbin_row_threshold

        sheet_name = "EDA - Interactions"
        worksheet = writer.book.add_worksheet(sheet_name)
        writer.sheets[sheet_name] = worksheet
        self._register_excel_sheet(writer, sheet_name)

        row = _EXCEL_NAV_ROW_OFFSET
        anchor_col_letter = _excel_col_letter(_EXCEL_NAV_COL_OFFSET)
        for i, c1 in enumerate(columns):
            for c2 in columns[i + 1 :]:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                if use_hexbin:
                    hexbin = ax.hexbin(data[c1], data[c2], gridsize=30, cmap="Blues")
                    fig.colorbar(hexbin, ax=ax, label="count")
                else:
                    sns.scatterplot(x=data[c1], y=data[c2], ax=ax, s=12, alpha=0.5)
                ax.set_title(f"{c1} vs {c2}")
                ax.set_xlabel(c1)
                ax.set_ylabel(c2)
                fig.tight_layout()

                image_data = io.BytesIO()
                fig.savefig(image_data, format="png", dpi=100)
                plt.close(fig)
                image_data.seek(0)
                worksheet.insert_image(
                    f"{anchor_col_letter}{row + 1}",
                    f"{c1}_vs_{c2}.png",
                    {"image_data": image_data},
                )
                row += 20

    def _numeric_histogram_data(self, series: pd.Series) -> tuple[pd.DataFrame, bool]:
        """Compute histogram bins (+ a count-scaled KDE curve, where computable).

        Thin delegate over ``html_widgets.compute_histogram`` — shared by
        ``to_excel()`` (native xlsxwriter chart) and ``to_html()`` (inline
        SVG), one histogram implementation for both output formats.

        Args:
            series (pd.Series): Clean (already dropna'd) numeric values.

        Returns:
            tuple[pd.DataFrame, bool]: See ``html_widgets.compute_histogram``.
        """
        return hw.compute_histogram(
            series,
            bins=settings.eda_histogram_bins,
            clip_lower_pct=self.excel_chart_clip_lower_pct,
            clip_upper_pct=self.excel_chart_clip_upper_pct,
        )

    def _write_numeric_charts_excel_sheet(self, writer: Any, numeric_df: pd.DataFrame) -> None:
        """Write the ``EDA - Numeric Charts`` sheet: one histogram per numeric feature.

        Each feature gets a small ``bin_center, count, kde`` table followed
        by a native column+line combo chart (histogram bars + a KDE curve
        where computable). Reads raw values via
        ``EDAReport.numeric_clean_series()`` — the EDA-time training data,
        not ``self.split.X_train`` (which may already reflect post-feature-
        selection columns by report-generation time).

        Args:
            writer (pd.ExcelWriter): Open writer using the ``xlsxwriter``
                engine.
            numeric_df (pd.DataFrame): Output of ``EDAReport.numeric_summary()``,
                supplying the column list and order.

        Returns:
            None
        """
        if numeric_df.empty:
            return
        sheet_name = "EDA - Numeric Charts"
        offset = _EXCEL_NAV_ROW_OFFSET
        col_offset = _EXCEL_NAV_COL_OFFSET
        bin_letter = _excel_col_letter(col_offset)
        count_letter = _excel_col_letter(col_offset + 1)
        kde_letter = _excel_col_letter(col_offset + 2)
        anchor_col_letter = _excel_col_letter(col_offset + 4)
        workbook = writer.book
        row = offset
        registered = False
        for feature in numeric_df["feature"]:
            series = self.eda_report.numeric_clean_series(feature)
            hist_df, has_kde = self._numeric_histogram_data(series)
            if hist_df.empty:
                continue
            hist_df.to_excel(
                writer, sheet_name=sheet_name, index=False, startrow=row, startcol=col_offset
            )
            if not registered:
                self._register_excel_sheet(writer, sheet_name)
                registered = True
            worksheet = writer.sheets[sheet_name]
            n_rows = len(hist_df)
            first_row = row + 2
            last_row = row + 1 + n_rows
            cat_range = f"${bin_letter}${first_row}:${bin_letter}${last_row}"
            count_range = f"${count_letter}${first_row}:${count_letter}${last_row}"
            column_chart = workbook.add_chart({"type": "column"})
            column_chart.add_series(
                {
                    "categories": f"='{sheet_name}'!{cat_range}",
                    "values": f"='{sheet_name}'!{count_range}",
                    "name": "count",
                }
            )
            if has_kde:
                line_chart = workbook.add_chart({"type": "line"})
                line_chart.add_series(
                    {
                        "categories": f"='{sheet_name}'!{cat_range}",
                        "values": (
                            f"='{sheet_name}'!${kde_letter}${first_row}:${kde_letter}${last_row}"
                        ),
                        "name": "kde",
                    }
                )
                column_chart.combine(line_chart)
            column_chart.set_title({"name": f"{feature} — distribution"})
            worksheet.insert_chart(f"{anchor_col_letter}{row + 1}", column_chart)
            row += max(n_rows, 15) + 2

    def _write_categorical_charts_excel_sheet(
        self, writer: Any, categorical_df: pd.DataFrame
    ) -> None:
        """Write the ``EDA - Categorical Charts`` sheet: one bar chart per
        categorical feature, per split.

        Adds a ``pct`` column next to ``count``, with one section per split
        (Train/Test/Validation/OOT), each holding one ``value, count, pct``
        table + bar chart per feature (top
        ``settings.eda_top_categorical_values`` values). Train reads via
        ``self.eda_report.categorical_value_counts()`` (EDA-time training
        data); Test/Validation/OOT are computed via a fresh
        ``UnivariateAnalyser`` fit on ``self.eda_split`` (the
        pre-feature-engineering split — not ``self.split``, which is
        already-encoded/numeric in a ``PipelineRunner`` run and would
        report zero categorical columns; same mechanism as
        ``_eda_summary_by_split()``), reusing Train's feature list so every
        split reports on the same columns.

        Args:
            writer (pd.ExcelWriter): Open writer using the ``xlsxwriter``
                engine.
            categorical_df (pd.DataFrame): Output of
                ``EDAReport.categorical_summary()``, supplying the Train
                feature list and order.

        Returns:
            None
        """
        if categorical_df.empty:
            return
        from types import SimpleNamespace

        from dscompanion.eda import UnivariateAnalyser

        sheet_name = "EDA - Categorical Charts"
        offset = _EXCEL_NAV_ROW_OFFSET
        col_offset = _EXCEL_NAV_COL_OFFSET
        value_letter = _excel_col_letter(col_offset)
        count_letter = _excel_col_letter(col_offset + 1)
        anchor_col_letter = _excel_col_letter(col_offset + 4)
        workbook = writer.book
        bold = writer.book.add_format({"bold": True})
        features = categorical_df["feature"].tolist()

        split_sources: dict[str, Any] = {"Train": self.eda_report}
        other_specs = [
            ("Test", self.eda_split.X_test),
            ("Validation", getattr(self.eda_split, "X_val", None)),
            ("OOT", getattr(self.eda_split, "X_oot", None)),
        ]
        for label, X in other_specs:
            if X is None or len(X) == 0:
                continue
            try:
                split_sources[label] = UnivariateAnalyser().fit(SimpleNamespace(train_X=X))
            except Exception as exc:
                logger.debug("EDA categorical charts (%s) skipped: %s", label, exc)

        row = offset
        registered = False
        for label, source in split_sources.items():
            vc_pairs = [(f, source.categorical_value_counts(f)) for f in features]
            vc_pairs = [(f, vc) for f, vc in vc_pairs if not vc.empty]
            if not vc_pairs:
                continue
            label_row = row
            row += 1
            for feature, vc in vc_pairs:
                counts = vc.to_numpy()
                pct = counts / counts.sum() * 100
                vc_df = pd.DataFrame({"value": vc.index.astype(str), "count": counts, "pct": pct})
                vc_df.to_excel(
                    writer, sheet_name=sheet_name, index=False, startrow=row, startcol=col_offset
                )
                if not registered:
                    self._register_excel_sheet(writer, sheet_name)
                    registered = True
                worksheet = writer.sheets[sheet_name]
                n_rows = len(vc_df)
                first_row = row + 2
                last_row = row + 1 + n_rows
                chart = workbook.add_chart({"type": "bar"})
                chart.add_series(
                    {
                        "categories": (
                            f"='{sheet_name}'!${value_letter}${first_row}:${value_letter}${last_row}"
                        ),
                        "values": (
                            f"='{sheet_name}'!${count_letter}${first_row}:${count_letter}${last_row}"
                        ),
                        "name": feature,
                    }
                )
                chart.set_title({"name": f"{feature} — top values ({label})"})
                worksheet.insert_chart(f"{anchor_col_letter}{row + 1}", chart)
                row += max(n_rows, 15) + 2
            worksheet.write(label_row, col_offset, label, bold)

    @staticmethod
    def _add_bar_chart(
        workbook: Any,
        worksheet: Any,
        sheet_name: str,
        n_rows: int,
        category_col: int,
        value_col: int,
        title: str,
        start_row: int = 0,
        anchor_col: str = _excel_col_letter(_EXCEL_NAV_COL_OFFSET + 5),
    ) -> None:
        """Insert a native xlsxwriter bar chart referencing an already-written column range.

        Args:
            workbook (Any): ``xlsxwriter.Workbook`` (``writer.book``).
            worksheet (Any): ``xlsxwriter.Worksheet`` the data was written to.
            sheet_name (str): Worksheet name, used to build quoted range
                references (sheet names contain spaces/hyphens).
            n_rows (int): Number of data rows (excludes the header row).
            category_col (int): 0-indexed column holding chart category labels.
            value_col (int): 0-indexed column holding chart values.
            title (str): Chart title.
            start_row (int): 0-indexed row of the header row for this table
                (nonzero only when a second table is stacked below another
                on the same sheet). Defaults to ``0``.
            anchor_col (str): Column letter to anchor the chart's top-left
                corner at. Defaults to ``"F"``.

        Returns:
            None
        """
        if n_rows == 0:
            return
        cat_letter = chr(ord("A") + category_col)
        val_letter = chr(ord("A") + value_col)
        first_data_row = start_row + 2
        last_data_row = start_row + 1 + n_rows
        cat_range = f"${cat_letter}${first_data_row}:${cat_letter}${last_data_row}"
        val_range = f"${val_letter}${first_data_row}:${val_letter}${last_data_row}"
        chart = workbook.add_chart({"type": "bar"})
        chart.add_series(
            {
                "categories": f"='{sheet_name}'!{cat_range}",
                "values": f"='{sheet_name}'!{val_range}",
                "name": title,
            }
        )
        chart.set_title({"name": title})
        worksheet.insert_chart(f"{anchor_col}{start_row + 1}", chart)

    @staticmethod
    def _add_decile_chart(
        workbook: Any,
        worksheet: Any,
        sheet_name: str,
        n_rows: int,
        start_row: int = 0,
        start_col: int = 0,
    ) -> None:
        """Insert a combo chart (event_rate columns + cumulative_lift line) on the decile sheet.

        Args:
            workbook (Any): ``xlsxwriter.Workbook`` (``writer.book``).
            worksheet (Any): ``xlsxwriter.Worksheet`` the decile table was
                written to.
            sheet_name (str): Worksheet name, used to build quoted range
                references.
            n_rows (int): Number of decile rows (excludes the header row).
            start_row (int): 0-indexed row of the table's header. Defaults
                to ``0``.
            start_col (int): 0-indexed column the table's first column
                (``decile``) was written to — every other column reference
                is computed relative to this. Defaults to ``0``.

        Returns:
            None
        """
        if n_rows == 0:
            return
        first_row = start_row + 2
        last_row = start_row + 1 + n_rows
        decile_letter = _excel_col_letter(start_col)
        event_rate_letter = _excel_col_letter(start_col + 3)
        cumulative_lift_letter = _excel_col_letter(start_col + 10)
        column_chart = workbook.add_chart({"type": "column"})
        column_chart.add_series(
            {
                "categories": (
                    f"='{sheet_name}'!${decile_letter}${first_row}:${decile_letter}${last_row}"
                ),
                "values": (
                    f"='{sheet_name}'!${event_rate_letter}${first_row}:"
                    f"${event_rate_letter}${last_row}"
                ),
                "name": "event_rate",
            }
        )
        line_chart = workbook.add_chart({"type": "line"})
        line_chart.add_series(
            {
                "categories": (
                    f"='{sheet_name}'!${decile_letter}${first_row}:${decile_letter}${last_row}"
                ),
                "values": (
                    f"='{sheet_name}'!${cumulative_lift_letter}${first_row}:"
                    f"${cumulative_lift_letter}${last_row}"
                ),
                "name": "cumulative_lift",
                "y2_axis": True,
            }
        )
        column_chart.combine(line_chart)
        column_chart.set_title({"name": "Decile Table — Event Rate & Cumulative Lift"})
        anchor_col_letter = _excel_col_letter(start_col + 12)
        worksheet.insert_chart(f"{anchor_col_letter}{start_row + 2}", column_chart)

    def to_tracking_artifact(self, run_id: str | None = None) -> None:
        """Export the model card to Excel and HTML (always), and Word
        (best-effort), then upload all produced files to the active
        tracking run under the ``model_card/`` subdirectory.

        Writes files to a temporary directory which is cleaned up
        automatically after upload. The Word export failure is caught and
        logged at DEBUG level; Excel and HTML are always uploaded.

        Must be called after ``generate()``.

        Args:
            run_id (str, optional): Tracking run ID to attach the artifacts
                to. Currently unused directly — artifacts are logged to
                whichever run is active in the calling context. Defaults
                to ``None``.

        Returns:
            None: This method always returns ``None``. Word export
            failures are swallowed and logged at DEBUG level.
        """
        import tempfile
        from pathlib import Path as _P

        from dscompanion.tracking import log_artifact

        with tempfile.TemporaryDirectory() as tmpdir:
            excel_path = self.to_excel(_P(tmpdir) / "model_card.xlsx")
            html_path = self.to_html(_P(tmpdir) / "model_card.html")
            try:
                word_path = self.to_word(_P(tmpdir) / "model_card.docx")
                log_artifact(str(word_path), artifact_path="model_card")
            except Exception as exc:
                logger.debug("Word export skipped: %s", exc)
            log_artifact(str(excel_path), artifact_path="model_card")
            log_artifact(str(html_path), artifact_path="model_card")

    def to_dict(self) -> dict[str, Any]:
        """Serialise all populated sections to a JSON-compatible Python dict,
        converting any ``pd.DataFrame`` values to a list of record dicts.

        Must be called after ``generate()``.

        Args:
            None

        Returns:
            Dict[str, Any]: Mapping of section key (str) to section data.
            ``pd.DataFrame`` values are converted to
            ``list[dict]`` via ``to_dict(orient="records")``; nested
            DataFrames inside sub-dicts are also converted.  Returns an
            empty dict when ``generate()`` has not yet been called.
        """
        result = {}
        for key, data in self.sections_.items():
            if isinstance(data, pd.DataFrame):
                result[key] = data.to_dict(orient="records")
            elif isinstance(data, dict):
                result[key] = {
                    k: v.to_dict(orient="records") if isinstance(v, pd.DataFrame) else v
                    for k, v in data.items()
                }
            else:
                result[key] = data
        return result

    def _config_summary_section(self) -> pd.DataFrame:
        """Build the full config-parameter table shown at the top of the report.

        Every monitored parameter is included — not just the ones that
        deviate — so a reviewer can verify "all defaults" by inspecting the
        table rather than trusting an absence of rows.

        Args:
            None

        Returns:
            pd.DataFrame: Columns ``parameter``, ``choices``, ``default``,
            ``user_choice``, ``is_deviation`` — one row per monitored
            parameter. Empty (correctly-columned) DataFrame when
            ``config_checks`` is ``None`` or empty.
        """
        if not self.config_checks:
            return pd.DataFrame(columns=_CONFIG_CHECK_COLUMNS)
        return pd.DataFrame(self.config_checks)

    def _data_lineage(self) -> dict[str, Any]:
        s = self.split
        info: dict[str, Any] = {
            "train_rows": len(s.X_train),
            "train_cols": s.X_train.shape[1],
        }
        if s.X_val is not None:
            info["val_rows"] = len(s.X_val)
        if s.X_oot is not None:
            info["oot_rows"] = len(s.X_oot)
        if hasattr(s, "metadata") and s.metadata is not None:
            info["date_range"] = (
                str(s.metadata.date_range) if hasattr(s.metadata, "date_range") else ""
            )
        return info

    def _feature_inventory(self) -> pd.DataFrame:
        rows = []
        for col in self.model.feature_names or self.split.X_train.columns:
            rows.append(
                {
                    "feature": col,
                    "dtype": (
                        str(self.split.X_train[col].dtype)
                        if col in self.split.X_train
                        else "unknown"
                    ),
                }
            )
        return pd.DataFrame(rows)

    def _performance_metrics(self) -> pd.DataFrame:
        """Build the Performance Metrics table: one row per metric, one column per split.

        Delegates to ``self.model.evaluate()`` for the underlying tidy
        long-format computation (train/test/val/oot — see
        ``BaseDSCompanionModel.evaluate()``), then pivots it for display. ``val``
        is computed by ``evaluate()`` and remains available to other
        consumers (hyperparameter tuning objectives), but is intentionally
        excluded from this report view, which shows only train/test/OOT.

        Args:
            None

        Returns:
            pd.DataFrame: Column ``metric`` plus one column per split present
            among ``train``/``test``/``oot`` (in that order; a split's column
            is omitted if that split wasn't evaluated, e.g. no OOT data).
            Empty DataFrame if ``evaluate()`` raises or returns no rows.
        """
        try:
            long_df = self.model.evaluate(self.split)
        except Exception as exc:
            logger.debug("Performance metrics skipped: %s", exc)
            return pd.DataFrame()
        if long_df.empty:
            return long_df
        display_splits = [s for s in ("train", "test", "oot") if s in long_df["split"].unique()]
        metric_order = long_df["metric"].drop_duplicates().tolist()
        wide = (
            long_df[long_df["split"].isin(display_splits)]
            .pivot(index="metric", columns="split", values="value")
            .reindex(index=metric_order, columns=display_splits)
            .rename(columns={"oot": "OOT"})
            .reset_index()
        )
        return wide

    def _comparison_metrics_table(self) -> pd.DataFrame:
        """Build a wide Train/Test/Validation/OOT comparison table, metric per row.

        A wide table (one row per metric, one column per split) makes a
        train/test/OOT performance gap visible by scanning a single row.
        Placed at the top of the Excel Performance Metrics sheet, above
        the diagnostics block. Unlike ``_performance_metrics()`` (Word/
        HTML/``to_dict()``, train/test/OOT only), this includes Validation
        too.

        Args:
            None

        Returns:
            pd.DataFrame: Column ``metric`` plus one column per split
            present among ``train``/``test``/``val``/``oot`` (renamed
            ``"Train"``/``"Test"``/``"Validation"``/``"OOT"``, in that
            order). Empty DataFrame if ``evaluate()`` raises or returns no
            rows.
        """
        try:
            long_df = self.model.evaluate(self.split)
        except Exception as exc:
            logger.debug("Comparison metrics table skipped: %s", exc)
            return pd.DataFrame()
        if long_df.empty:
            return long_df
        rename = {"train": "Train", "test": "Test", "val": "Validation", "oot": "OOT"}
        display_splits = [s for s in rename if s in long_df["split"].unique()]
        metric_order = long_df["metric"].drop_duplicates().tolist()
        wide = (
            long_df[long_df["split"].isin(display_splits)]
            .pivot(index="metric", columns="split", values="value")
            .reindex(index=metric_order, columns=display_splits)
            .rename(columns=rename)
            .reset_index()
        )
        return wide

    def _performance_diagnostics(self) -> list[dict[str, str]]:
        """Flag suspicious train/test/OOT performance patterns for the Excel sheet.

        Deliberately duplicated from ``dscompanion.api.services.evaluate``'s
        ``_suspicious_flags`` (itself already duplicated from
        ``interactive_step_evaluate.py`` — this is the third copy, matching
        that existing convention) rather than imported: ``dscompanion.docs`` is a
        core, always-shipped module, while ``dscompanion.api`` is an optional
        extra (``pip install dscompanion[api]``) — importing it here would
        break ``ModelCard`` wherever ``dscompanion.api``'s dependencies
        aren't installed. Uses the same ``settings.evaluate_*`` thresholds so
        the two independent implementations stay calibrated the same way.

        Args:
            None

        Returns:
            list[dict[str, str]]: One ``{"level": ..., "message": ...}`` dict
            per suspicious pattern found (``level`` one of ``"error"``,
            ``"warning"``, ``"info"``). Empty list when metrics look healthy
            or ``evaluate()`` fails.
        """
        try:
            metrics_df = self.model.evaluate(self.split)
        except Exception as exc:
            logger.debug("Performance diagnostics skipped: %s", exc)
            return []
        if metrics_df.empty:
            return []

        def _value(split: str, metric: str) -> float | None:
            row = metrics_df[(metrics_df["split"] == split) & (metrics_df["metric"] == metric)]
            if row.empty:
                return None
            val = float(row["value"].iloc[0])
            return None if val != val else val  # NaN check without importing math/numpy here

        primary = next(
            (s for s in ("val", "test", "train") if (metrics_df["split"] == s).any()), "train"
        )
        primary_auc = _value(primary, "roc_auc")
        train_auc = _value("train", "roc_auc")
        flags: list[dict[str, str]] = []

        if primary_auc is not None:
            if primary_auc > settings.evaluate_leakage_auc_threshold:
                flags.append(
                    {
                        "level": "error",
                        "message": (
                            f"AUC of {primary_auc:.3f} on {primary} is unusually high — "
                            "possible data leakage (a feature accidentally contains the answer)."
                        ),
                    }
                )
            elif primary_auc < settings.evaluate_near_random_auc_threshold:
                flags.append(
                    {
                        "level": "warning",
                        "message": (
                            f"AUC of {primary_auc:.3f} on {primary} is close to random "
                            "guessing (0.50) — the model may not have learned anything useful."
                        ),
                    }
                )

        if train_auc is not None and primary_auc is not None and primary != "train":
            gap = train_auc - primary_auc
            if gap > settings.evaluate_overfitting_gap_threshold:
                flags.append(
                    {
                        "level": "warning",
                        "message": (
                            f"Training AUC ({train_auc:.3f}) is much higher than {primary} "
                            f"AUC ({primary_auc:.3f}), a gap of {gap:.3f} — the model may have "
                            "overfit to the training data."
                        ),
                    }
                )

        primary_psi = _value(primary, "psi")
        if primary_psi is not None:
            if primary_psi > settings.evaluate_psi_unstable_threshold:
                flags.append(
                    {
                        "level": "warning",
                        "message": (
                            f"PSI of {primary_psi:.3f} on {primary} is high (> "
                            f"{settings.evaluate_psi_unstable_threshold}) — the score "
                            "distribution shifted significantly vs. training."
                        ),
                    }
                )
            elif primary_psi > settings.evaluate_psi_monitor_threshold:
                flags.append(
                    {
                        "level": "info",
                        "message": (
                            f"PSI of {primary_psi:.3f} on {primary} is in the monitoring zone "
                            f"({settings.evaluate_psi_monitor_threshold}-"
                            f"{settings.evaluate_psi_unstable_threshold})."
                        ),
                    }
                )

        return flags

    def _decile_table_section(self) -> pd.DataFrame:
        """Build the 10-decile gains/lift table on the test split.

        Classification only — skipped when ``self.decile_table=False`` or
        when the model has no ``predict_proba`` (regression, clustering).

        Args:
            None

        Returns:
            pd.DataFrame: See ``dscompanion.utils.metrics.decile_table`` for the
            column schema. Empty (correctly-columned) DataFrame when the
            decile table is disabled, not applicable, or computation fails.
        """
        from dscompanion.utils.metrics import DECILE_TABLE_COLUMNS, decile_table

        if not self.decile_table:
            return pd.DataFrame(columns=DECILE_TABLE_COLUMNS)
        try:
            y_prob = self.model.predict_proba(self.split.X_test)[:, 1]
            return decile_table(self.split.y_test.to_numpy(), y_prob)
        except Exception as exc:
            logger.debug("Decile table skipped: %s", exc)
            return pd.DataFrame(columns=DECILE_TABLE_COLUMNS)

    def _decile_table_by_split(self) -> dict[str, pd.DataFrame]:
        """Build the 10-decile gains/lift table for every available split.

        Unlike ``_decile_table_section()`` (test split only, used by
        Word/HTML/``to_dict()``), this includes Train/Validation/OOT too —
        so a large train-vs-test-vs-OOT gap in event rate or cumulative
        lift is visible directly, not just inferable from headline metrics.
        Used only by ``to_excel()``'s Decile Table sheet
        (``_write_decile_table_excel_sheet``), which renders each entry as
        its own labelled, charted, stacked section. Word, HTML, and
        ``to_dict()`` continue to use ``_decile_table_section()``'s
        original test-only shape, unchanged.

        Args:
            None

        Returns:
            dict[str, pd.DataFrame]: Keyed by ``"Train"``, ``"Test"``,
            ``"Validation"``, ``"OOT"`` (only splits with data are
            included, in that order), each with
            ``dscompanion.utils.metrics.DECILE_TABLE_COLUMNS`` columns. Empty
            dict when ``self.decile_table=False`` or the model has no
            usable ``predict_proba`` on any split.
        """
        from dscompanion.utils.metrics import decile_table

        if not self.decile_table:
            return {}

        split_specs = [
            ("Train", self.split.X_train, self.split.y_train),
            ("Test", self.split.X_test, self.split.y_test),
            ("Validation", getattr(self.split, "X_val", None), getattr(self.split, "y_val", None)),
            ("OOT", getattr(self.split, "X_oot", None), getattr(self.split, "y_oot", None)),
        ]
        result: dict[str, pd.DataFrame] = {}
        for label, X, y in split_specs:
            if X is None or len(X) == 0:
                continue
            try:
                y_prob = self.model.predict_proba(X)[:, 1]
                result[label] = decile_table(y.to_numpy(), y_prob)
            except Exception as exc:
                logger.debug("Decile table (%s) skipped: %s", label, exc)
        return result

    def _stability(self) -> dict[str, Any]:
        """Compute per-feature PSI between train and OOT splits.

        Uses ``feature_psi_table()`` from ``dscompanion.utils`` to compute PSI for
        every shared column between ``X_train`` and ``X_oot``.  Returns an empty
        note when no OOT split is available.

        Args:
            None

        Returns:
            dict: Contains ``feature_psi`` (list of per-feature records with
            ``feature``, ``psi``, and ``flag`` keys), ``flagged_count`` (int),
            and ``total_features`` (int).  Returns ``{"note": "..."}`` when no
            OOT data is available or ``feature_psi_table`` raises.
        """
        s = self.split
        if s.X_oot is None or len(s.X_oot) == 0:
            return {"note": "No OOT split available — feature PSI not computed."}

        try:
            from dscompanion.config import settings
            from dscompanion.utils import feature_psi_table

            psi_df = feature_psi_table(s.X_train, s.X_oot)
            flagged = int((psi_df["psi"] > settings.psi_alert_threshold).sum())
            return {
                "feature_psi": psi_df.to_dict(orient="records"),
                "flagged_count": flagged,
                "total_features": len(psi_df),
            }
        except Exception as exc:
            logger.debug("Feature PSI skipped: %s", exc)
            return {"note": "Feature PSI computation failed — see debug log."}

    def _calibration_section(self) -> dict[str, Any]:
        if self.calibrator is None:
            return {"note": "No calibrator provided."}
        try:
            report = self.calibrator.calibration_report()
            return report.to_dict(orient="records")[0]
        except Exception:
            return {}

    def _explainability(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.explainer is not None:
            try:
                top = self.explainer.mean_abs_shap().head(20)
                result["top_features"] = top.to_dict(orient="records")
            except Exception:
                pass
        if self.permutation_importance is not None:
            try:
                top = self.permutation_importance.importance_table().head(20)
                result["permutation_top_features"] = top.to_dict(orient="records")
            except Exception:
                pass
        if not result:
            return {"note": "No SHAP explainer or permutation importance provided."}
        return result

    def _tuning_section(self) -> dict[str, Any]:
        if self.tuner is None:
            return {"note": "No tuner provided."}
        result: dict[str, Any] = {}
        try:
            result["backend"] = self.tuner.backend
            result["n_trials"] = self.tuner.n_trials
            result["metric"] = self.tuner.metric
            result["best_params"] = self.tuner.best_params_
            result["best_score"] = self.tuner.best_score_
        except Exception:
            pass
        return result

    def _leaderboard_section(self) -> pd.DataFrame:
        """Return the multi-algorithm comparison table from a ``Leaderboard`` run, if any.

        Populated when ``PipelineConfig.leaderboard.enabled=True`` drove
        this run — ``PipelineRunner`` threads the fitted ``Leaderboard``
        instance through to this card's constructor.

        Args:
            None

        Returns:
            pd.DataFrame: ``self.leaderboard.leaderboard_`` when a
            ``Leaderboard`` instance was supplied and already run. Empty
            (correctly-columned) DataFrame otherwise — section is never
            skipped, matching ``decile_table``'s convention.
        """
        empty_cols = ["algorithm", "status", "fit_time_seconds", "error"]
        if self.leaderboard is None or self.leaderboard.leaderboard_ is None:
            return pd.DataFrame(columns=empty_cols)
        return self.leaderboard.leaderboard_

    def _leaderboard_metrics_by_split(self) -> dict[str, pd.DataFrame]:
        """Build a per-algorithm metrics table for every split, Excel-only (R.10).

        Reuses ``Leaderboard.fitted_models_`` (every algorithm that trained
        successfully, already fitted — no refitting) and calls each
        model's own ``evaluate(self.split)`` to get its full per-split
        metrics, independent of whichever single ``eval_split``
        ``Leaderboard.leaderboard_`` was ranked on.

        Args:
            None

        Returns:
            dict[str, pd.DataFrame]: Keyed by ``"Train"``, ``"Test"``,
            ``"Validation"``, ``"OOT"`` (only splits with data are
            included, in that order). Each DataFrame has one row per
            successfully-fitted algorithm, columns ``algorithm`` plus every
            metric that algorithm's ``evaluate()`` returned for that split.
            Empty dict when no ``Leaderboard`` was supplied or nothing
            fitted successfully.
        """
        if self.leaderboard is None or not self.leaderboard.fitted_models_:
            return {}
        split_labels = {"train": "Train", "test": "Test", "val": "Validation", "oot": "OOT"}
        rows_by_split: dict[str, list[dict[str, Any]]] = {
            label: [] for label in split_labels.values()
        }
        for algorithm, model in self.leaderboard.fitted_models_.items():
            try:
                metrics_df = model.evaluate(self.split)
            except Exception as exc:
                logger.debug("Leaderboard per-split metrics (%s) skipped: %s", algorithm, exc)
                continue
            for key, label in split_labels.items():
                split_metrics = metrics_df[metrics_df["split"] == key]
                if split_metrics.empty:
                    continue
                row: dict[str, Any] = {"algorithm": algorithm}
                row.update(dict(zip(split_metrics["metric"], split_metrics["value"])))
                rows_by_split[label].append(row)
        return {label: pd.DataFrame(rows) for label, rows in rows_by_split.items() if rows}

    def _full_config_section(self) -> dict[str, Any]:
        """Build the verbatim experiment-config section shown at the bottom of the report.

        Unlike ``config_checks`` (which only covers the monitored parameter
        subset), this section publishes the complete YAML actually used for
        the run — the reviewer's ground truth for every parameter, including
        ones dscompanion doesn't track as a behavioural choice.

        Args:
            None

        Returns:
            dict[str, Any]: ``{"yaml": self.raw_config_yaml}`` when set, else
            ``{"note": "..."}``.
        """
        if not self.raw_config_yaml:
            return {"note": "No experiment config was captured for this run."}
        return {"yaml": self.raw_config_yaml}
