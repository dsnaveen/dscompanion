"""Tests for dscompanion.docs.html_model — non-EDA ModelCard section HTML renderers."""

from __future__ import annotations

import pandas as pd

from dscompanion.docs import html_model


def test_render_performance_section_empty_dataframe_shows_note():
    html = html_model.render_performance_section(pd.DataFrame())
    assert "No performance metrics" in html


def test_render_performance_section_renders_metric_rows():
    df = pd.DataFrame({"metric": ["gini", "roc_auc"], "train": [0.6, 0.8], "test": [0.55, 0.77]})
    html = html_model.render_performance_section(df)
    assert "gini" in html
    assert "roc_auc" in html
    assert "0.6" in html


def test_render_decile_section_empty_dataframe_shows_note():
    from dscompanion.utils.metrics import DECILE_TABLE_COLUMNS

    html = html_model.render_decile_section(pd.DataFrame(columns=DECILE_TABLE_COLUMNS))
    assert "No decile table" in html


def test_render_decile_section_renders_combo_chart():
    df = pd.DataFrame(
        {
            "decile": [1, 2, 3],
            "count": [10, 10, 10],
            "events": [8, 5, 2],
            "event_rate": [0.8, 0.5, 0.2],
            "cumulative_count": [10, 20, 30],
            "cumulative_events": [8, 13, 15],
            "cumulative_event_rate": [0.8, 0.65, 0.5],
            "pct_of_total_events": [0.53, 0.33, 0.13],
            "cumulative_pct_of_total_events": [0.53, 0.86, 1.0],
            "lift": [1.6, 1.0, 0.4],
            "cumulative_lift": [1.6, 1.3, 1.0],
        }
    )
    html = html_model.render_decile_section(df)
    assert "<svg" in html
    assert "<table" in html


def test_render_calibration_section_note_when_no_calibrator():
    html = html_model.render_calibration_section({"note": "No calibrator provided."})
    assert "No calibrator provided." in html


def test_render_calibration_section_renders_ece_before_after():
    data = {"method": "isotonic", "ece_before": 0.12, "ece_after": 0.03}
    html = html_model.render_calibration_section(data)
    assert "isotonic" in html
    assert "0.12" in html
    assert "0.03" in html


def test_render_explainability_section_note_when_no_explainer():
    html = html_model.render_explainability_section({"note": "No SHAP explainer provided."})
    assert "No SHAP explainer provided." in html


def test_render_explainability_section_renders_bar_chart():
    data = {
        "top_features": [
            {"feature": "income", "mean_abs_shap": 0.5, "rank": 1},
            {"feature": "balance", "mean_abs_shap": 0.2, "rank": 2},
        ]
    }
    html = html_model.render_explainability_section(data)
    assert "<svg" in html
    assert "income" in html


def test_render_explainability_section_renders_permutation_table():
    data = {
        "permutation_top_features": [
            {"feature": "income", "importance_mean": 0.4, "importance_std": 0.01, "rank": 1},
            {"feature": "balance", "importance_mean": 0.1, "importance_std": 0.02, "rank": 2},
        ]
    }
    html = html_model.render_explainability_section(data)
    assert "<svg" in html
    assert "income" in html


def test_render_explainability_section_renders_both_shap_and_permutation():
    data = {
        "top_features": [{"feature": "income", "mean_abs_shap": 0.5, "rank": 1}],
        "permutation_top_features": [
            {"feature": "balance", "importance_mean": 0.1, "importance_std": 0.02, "rank": 1}
        ],
    }
    html = html_model.render_explainability_section(data)
    assert html.count("<svg") == 2
    assert "income" in html
    assert "balance" in html


def test_render_stability_section_note_when_no_oot():
    html = html_model.render_stability_section(
        {"note": "No OOT split available — feature PSI not computed."}
    )
    assert "No OOT split available" in html


def test_render_stability_section_renders_flagged_count():
    data = {
        "feature_psi": [{"feature": "income", "psi": 0.3, "flag": True}],
        "flagged_count": 1,
        "total_features": 1,
    }
    html = html_model.render_stability_section(data)
    assert "1 / 1" in html


def test_render_tuning_section_note_when_no_tuner():
    html = html_model.render_tuning_section({"note": "No tuner provided."})
    assert "No tuner provided." in html


def test_render_tuning_section_renders_best_params():
    data = {
        "backend": "optuna",
        "n_trials": 50,
        "metric": "gini",
        "best_params": {"max_depth": 5, "learning_rate": 0.1},
        "best_score": 0.62,
    }
    html = html_model.render_tuning_section(data)
    assert "optuna" in html
    assert "max_depth" in html


def test_render_leaderboard_section_empty_shows_note():
    html = html_model.render_leaderboard_section(
        pd.DataFrame(columns=["algorithm", "status", "fit_time_seconds", "error"])
    )
    assert "not enabled" in html


def test_render_leaderboard_section_renders_ranked_rows():
    df = pd.DataFrame({"algorithm": ["xgboost", "lightgbm"], "status": ["ok", "ok"]})
    html = html_model.render_leaderboard_section(df)
    assert "xgboost" in html
    assert "lightgbm" in html


def test_render_feature_inventory_section_renders_all_features():
    df = pd.DataFrame({"feature": ["a", "b"], "dtype": ["float64", "object"]})
    html = html_model.render_feature_inventory_section(df)
    assert "a" in html
    assert "b" in html


def test_render_full_config_section_note_when_no_yaml():
    html = html_model.render_full_config_section(
        {"note": "No experiment config was captured for this run."}
    )
    assert "No experiment config" in html


def test_render_full_config_section_renders_yaml_verbatim():
    html = html_model.render_full_config_section({"yaml": "name: test_experiment\n"})
    assert "test_experiment" in html


def test_render_governance_section_includes_sign_off_roles_and_limitations():
    governance_df = pd.DataFrame([{"role": "Author", "name": "N", "date": "", "sign_off": ""}])
    limitations = {"notes": "", "boilerplate": "This model was trained on historical data."}
    html = html_model.render_governance_section(governance_df, limitations)
    assert "Author" in html
    assert "historical data" in html


def test_render_config_summary_section_empty_shows_note():
    from dscompanion.docs.model_card import _CONFIG_CHECK_COLUMNS

    html = html_model.render_config_summary_section(pd.DataFrame(columns=_CONFIG_CHECK_COLUMNS))
    assert "No config deviation checks" in html


def test_render_config_summary_section_renders_deviation_rows():
    df = pd.DataFrame(
        [
            {
                "parameter": "eda.enabled",
                "choices": "True / False",
                "default": True,
                "user_choice": False,
                "is_deviation": True,
            },
        ]
    )
    html = html_model.render_config_summary_section(df)
    assert "eda.enabled" in html


def test_render_overview_section_shows_deviation_banner():
    model_summary = {"algorithm": "XGBClassifier", "version": "1.0"}
    data_lineage = {"train_rows": 1000, "train_cols": 10}
    config_summary = pd.DataFrame(
        [
            {
                "parameter": "eda.enabled",
                "choices": "True / False",
                "default": True,
                "user_choice": False,
                "is_deviation": True,
            },
        ]
    )
    html = html_model.render_overview_section(model_summary, data_lineage, config_summary)
    assert "1 non-default config choice" in html
    assert "XGBClassifier" in html


def test_render_overview_section_shows_clean_banner_when_no_deviations():
    html = html_model.render_overview_section({}, {}, pd.DataFrame())
    assert "No non-default config choices." in html
