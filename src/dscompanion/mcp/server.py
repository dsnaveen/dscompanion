"""FastMCP server exposing dscompanion as 5 atomic, agent-callable tools.

Runs entirely locally (stdio transport, launched by the user's own MCP client
as a subprocess) -- no data leaves the local machine, no network calls, no
auth. Every tool returns a plain JSON-serialisable dict, never a DataFrame or
other Python object, and never lets an exception propagate as a raw
traceback -- callers get a structured ``{"error": str}`` instead, since an
agent needs a parseable failure it can explain to the user.
"""

# Deliberately no `from __future__ import annotations` here, unlike every other
# module in this codebase -- the installed `mcp` SDK's @mcp.tool() decorator
# introspects live parameter type objects at import time (Tool.from_function calls
# issubclass(param.annotation, Context)), and postponed evaluation makes annotations
# strings instead, crashing with "issubclass() arg 1 must be a class". Confirmed via
# a minimal reproduction against mcp==1.12.4. Safe to omit since this project targets
# Python 3.12+, where modern generic/union syntax (`dict[str, Any]`, `X | None`)
# already works natively without the future import.

import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from dscompanion.config import settings
from dscompanion.eda import EDAReport
from dscompanion.explain import SHAPExplainer
from dscompanion.leaderboard import Leaderboard
from dscompanion.mcp._artifacts import load_split, new_run_dir, save_split
from dscompanion.mcp._loading import load_dataframe
from dscompanion.mcp._readiness import readiness_checks
from dscompanion.models.base import BaseDSCompanionModel
from dscompanion.split import DataSplitter

logger = logging.getLogger(__name__)

__all__ = [
    "mcp",
    "analyze_dataset",
    "train_and_compare_models",
    "check_model_readiness",
    "explain_model",
]

mcp = FastMCP("dscompanion")


@mcp.tool()
def analyze_dataset(data_path: str, target: str) -> dict:
    """Run exploratory data analysis on a local dataset file.

    Use this first, before training a model, to understand a dataset's shape,
    quality, and its features' relationship to the target -- surfaces missing
    data, low-signal columns, and other red flags.

    Args:
        data_path (str): Path to a local CSV or Parquet file.
        target (str): Name of the target column in that file.

    Returns:
        dict: On success: ``{"run_dir": str, "summary": {"n_rows": int,
        "n_columns": int, "numeric_columns": list[str], "categorical_columns":
        list[str], "missing_pct_by_column": dict[str, float], "top_iv_features":
        list[dict], "red_flags": list[str]}, "chart_paths": list[str]}``. On
        failure: ``{"error": str}``.
    """
    try:
        df = load_dataframe(data_path)
        if target not in df.columns:
            return {"error": f"target column {target!r} not found in {data_path}"}

        split = DataSplitter(strategy="random", target_col=target).fit_split(df)
        report = EDAReport(split, target=target).run_all()

        run_dir = new_run_dir()
        chart_paths = report.export_charts(run_dir / "eda")

        numeric_summary = report.numeric_summary()
        categorical_summary = report.categorical_summary()
        iv_df = report.iv_table()

        missing_pct: dict[str, float] = {}
        for summary_df in (numeric_summary, categorical_summary):
            for _, row in summary_df.iterrows():
                missing_pct[row["feature"]] = float(row["missing_pct"])

        red_flags = [
            f"{feature}: {pct:.1%} missing"
            for feature, pct in missing_pct.items()
            if pct > settings.mcp_high_missing_red_flag_threshold
        ]

        summary = {
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "numeric_columns": numeric_summary["feature"].tolist(),
            "categorical_columns": categorical_summary["feature"].tolist(),
            "missing_pct_by_column": missing_pct,
            "top_iv_features": iv_df.head(10).to_dict("records"),
            "red_flags": red_flags,
        }
        return {
            "run_dir": str(run_dir),
            "summary": summary,
            "chart_paths": [str(p) for p in chart_paths.values()],
        }
    except Exception as exc:
        logger.warning("analyze_dataset failed: %s", exc)
        return {"error": str(exc)}


@mcp.tool()
def train_and_compare_models(data_path: str, target: str, task: str) -> dict:
    """Train and compare multiple algorithms on a dataset, returning the
    best-performing model.

    Use this after analyzing a dataset, to find the best model for it.
    Currently only task="classification" is supported.

    Args:
        data_path (str): Path to a local CSV or Parquet file.
        target (str): Name of the target column.
        task (str): One of "classification", "regression", "clustering". Only
            "classification" is currently implemented -- other values return
            a structured error rather than raising.

    Returns:
        dict: On success: ``{"run_dir": str, "model_path": str, "leaderboard":
        list[dict], "recommended": str}``. On failure: ``{"error": str}``.
    """
    if task != "classification":
        return {
            "error": (
                f"task={task!r} is not yet supported -- only 'classification' "
                "is currently implemented"
            )
        }
    try:
        df = load_dataframe(data_path)
        if target not in df.columns:
            return {"error": f"target column {target!r} not found in {data_path}"}

        split = DataSplitter(strategy="random", target_col=target).fit_split(df)
        leaderboard = Leaderboard(task=task)
        leaderboard_df = leaderboard.run(split)

        run_dir = new_run_dir()
        recommended = leaderboard.best_algorithm()
        model_path = leaderboard.fitted_models_[recommended].save(run_dir / "model.joblib")
        save_split(split, run_dir / "split.joblib")

        return {
            "run_dir": str(run_dir),
            "model_path": str(model_path),
            "leaderboard": leaderboard_df.to_dict("records"),
            "recommended": str(recommended),
        }
    except Exception as exc:
        logger.warning("train_and_compare_models failed: %s", exc)
        return {"error": str(exc)}


@mcp.tool()
def check_model_readiness(run_dir: str) -> dict:
    """Check whether a trained model is ready for production deployment.

    Use this after training a model (via train_and_compare_models), before
    deploying it, to check for overfitting, instability, data leakage, and
    near-random performance.

    Args:
        run_dir (str): The run directory returned by ``train_and_compare_models``
            -- must contain ``model.joblib`` and ``split.joblib``.

    Returns:
        dict: On success: ``{"checks": list[dict], "overall": str}`` where each
        check dict is ``{"name": str, "status": "pass"|"warn"|"fail", "detail":
        str}`` and ``overall`` is ``"ready"``, ``"needs_review"``, or
        ``"not_ready"``. On failure: ``{"error": str}``.
    """
    try:
        model_path = Path(run_dir) / "model.joblib"
        split_path = Path(run_dir) / "split.joblib"
        if not model_path.exists() or not split_path.exists():
            return {
                "error": (
                    f"{run_dir!r} does not contain both model.joblib and "
                    "split.joblib -- pass a run_dir returned by train_and_compare_models"
                )
            }

        model = BaseDSCompanionModel.load(model_path)
        split = load_split(split_path)
        checks = readiness_checks(model, split)

        statuses = {c["status"] for c in checks}
        if "fail" in statuses:
            overall = "not_ready"
        elif "warn" in statuses:
            overall = "needs_review"
        else:
            overall = "ready"

        return {"checks": checks, "overall": overall}
    except Exception as exc:
        logger.warning("check_model_readiness failed: %s", exc)
        return {"error": str(exc)}


@mcp.tool()
def explain_model(run_dir: str) -> dict:
    """Explain which features drive a trained model's predictions.

    Use this after training a model (via train_and_compare_models) to
    understand its behavior via SHAP-based global feature importance.

    Args:
        run_dir (str): The run directory returned by ``train_and_compare_models``
            -- must contain ``model.joblib`` and ``split.joblib``.

    Returns:
        dict: On success: ``{"top_features": list[dict]}`` where each entry has
        ``feature`` (str), ``mean_abs_shap`` (float), and ``rank`` (int). On
        failure: ``{"error": str}``.
    """
    try:
        model_path = Path(run_dir) / "model.joblib"
        split_path = Path(run_dir) / "split.joblib"
        if not model_path.exists() or not split_path.exists():
            return {
                "error": (
                    f"{run_dir!r} does not contain both model.joblib and "
                    "split.joblib -- pass a run_dir returned by train_and_compare_models"
                )
            }

        model = BaseDSCompanionModel.load(model_path)
        split = load_split(split_path)

        explainer = SHAPExplainer(model).fit(split.train_X)
        importance_df = explainer.mean_abs_shap()

        return {"top_features": importance_df.to_dict("records")}
    except Exception as exc:
        logger.warning("explain_model failed: %s", exc)
        return {"error": str(exc)}
