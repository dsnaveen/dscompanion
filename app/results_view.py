"""Post-run results rendering: metrics, leaderboard, decile table, SHAP, downloads.

Local-dev Streamlit UI only — never imported by dscompanion's runtime code.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from interactive_state import dark_fig

logger = logging.getLogger(__name__)

__all__ = ["render_results"]


def _render_metrics_tab(result: Any) -> None:
    st.caption(f"Elapsed: {result.elapsed_seconds:.1f}s · Run ID: {result.run_id or 'n/a'}")
    st.dataframe(result.metrics)


def _render_leaderboard_tab(result: Any) -> None:
    if result.config.leaderboard.enabled and result.leaderboard is not None:
        st.dataframe(result.leaderboard.leaderboard_)
        st.caption(f"Winning algorithm: {result.config.model.algorithm}")
    else:
        st.info("Leaderboard mode was not enabled for this run.")


def _render_decile_tab(result: Any) -> None:
    decile_df = result.model_card.sections_.get("decile_table") if result.model_card else None
    if decile_df is not None and len(decile_df) > 0:
        st.dataframe(decile_df)
    else:
        st.info(
            "No decile table available for this run (disabled, regression/clustering task, "
            "or the model has no predict_proba)."
        )


def _render_explainability_tab(result: Any) -> None:
    if result.config.explain.shap_enabled and result.explainer is not None:
        try:
            fig = result.explainer.summary_plot()
            # theme=None: render dscompanion's own apply_dscompanion_theme() styling as
            # authored — Streamlit's default theme="streamlit" re-skins
            # fonts/colors to match the page theme (dark mode washes out
            # dscompanion's dark-on-white text to near-invisible light grey).
            st.plotly_chart(dark_fig(fig), width="stretch", theme=None)
        except Exception as exc:
            logger.warning("SHAP summary plot failed: %s", type(exc).__name__)
            st.dataframe(result.explainer.mean_abs_shap())
    else:
        st.info("SHAP was not enabled for this run.")


def _render_downloads_tab(result: Any) -> None:
    deviations = result.config_deviations
    if deviations:
        st.warning("Non-default config choices:")
        for d in deviations:
            st.write(f"- **{d['parameter']}** = {d['user_choice']} (default: {d['default']})")
    else:
        st.success("No non-default config choices.")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        yaml_path = tmp_path / "config.yaml"
        result.config.to_yaml(yaml_path)
        st.download_button(
            "Download config (YAML)",
            yaml_path.read_bytes(),
            file_name=f"{result.config.name}_config.yaml",
            mime="text/yaml",
        )

        if result.report_path is not None and Path(result.report_path).exists():
            st.download_button(
                "Download HTML report",
                Path(result.report_path).read_bytes(),
                file_name=f"{result.config.name}_model_card.html",
                mime="text/html",
            )
        else:
            st.caption("HTML report not available (reporting.html_report was disabled).")

        if result.model_card is not None:
            try:
                xlsx_path = result.model_card.to_excel(tmp_path / "report.xlsx")
                st.download_button(
                    "Download Excel report",
                    xlsx_path.read_bytes(),
                    file_name=f"{result.config.name}_model_card.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception as exc:
                logger.warning("Excel report generation failed: %s", type(exc).__name__)
                st.caption("Excel report could not be generated for this run.")


def render_results(result: Any) -> None:
    """Render the post-run results area as a tabbed layout.

    Args:
        result (PipelineRunResult): The completed pipeline run result.

    Returns:
        None
    """
    tabs = st.tabs(
        ["Metrics", "Leaderboard", "Decile Table", "Explainability", "Config & Downloads"]
    )
    with tabs[0]:
        _render_metrics_tab(result)
    with tabs[1]:
        _render_leaderboard_tab(result)
    with tabs[2]:
        _render_decile_tab(result)
    with tabs[3]:
        _render_explainability_tab(result)
    with tabs[4]:
        _render_downloads_tab(result)
