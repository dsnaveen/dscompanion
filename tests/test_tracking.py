"""Tests for dscompanion.tracking.run_context — the local logging-based run tracker."""

from __future__ import annotations

import logging

from dscompanion.tracking.run_context import (
    RunHandle,
    _NoopRun,
    log_artifact,
    log_metrics,
    log_params,
    tracking_run,
)


class TestTrackingRun:
    def test_yields_noop_when_disabled(self):
        with tracking_run("test_run", enabled=False) as run:
            assert isinstance(run, _NoopRun)
            assert run.info.run_id == "noop-run-id"

    def test_disabled_emits_zero_log_lines(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="dscompanion.tracking.run_context"):
            with tracking_run("test_run", enabled=False):
                pass
        assert caplog.records == []

    def test_noop_run_has_experiment_id(self):
        noop = _NoopRun()
        assert noop.info.experiment_id == "0"

    def test_enabled_yields_run_handle_with_fresh_run_id(self):
        with tracking_run("my_run") as run:
            assert isinstance(run, RunHandle)
            assert isinstance(run.info.run_id, str)
            assert len(run.info.run_id) == 32  # uuid4().hex
            assert run.info.experiment_id == "0"

    def test_two_runs_get_different_run_ids(self):
        with tracking_run("run_a") as a:
            id_a = a.info.run_id
        with tracking_run("run_b") as b:
            id_b = b.info.run_id
        assert id_a != id_b

    def test_run_id_override_is_used_verbatim(self):
        with tracking_run("run_a", run_id="caller-supplied-id") as run:
            assert run.info.run_id == "caller-supplied-id"

    def test_run_id_none_still_generates_fresh_id(self):
        with tracking_run("run_a", run_id=None) as run:
            assert len(run.info.run_id) == 32  # uuid4().hex, unchanged default behavior

    def test_logs_start_and_finish_at_info(self, caplog):
        with caplog.at_level(logging.INFO, logger="dscompanion.tracking.run_context"):
            with tracking_run("my_run", tags={"owner": "your_name"}) as run:
                pass
        messages = [r.message for r in caplog.records]
        assert any("started" in m and run.info.run_id[:8] in m for m in messages)
        assert any("finished" in m and run.info.run_id[:8] in m for m in messages)

    def test_run_name_and_tags_appear_in_start_log(self, caplog):
        with caplog.at_level(logging.INFO, logger="dscompanion.tracking.run_context"):
            with tracking_run("my_model_v1", tags={"owner": "your_name"}):
                pass
        assert any("my_model_v1" in r.message and "your_name" in r.message for r in caplog.records)


class TestLogMetrics:
    def test_logs_one_line_per_metric(self, caplog):
        with caplog.at_level(logging.INFO, logger="dscompanion.tracking.run_context"):
            log_metrics({"roc_auc": 0.85, "ks": 0.42}, step=1)
        messages = [r.message for r in caplog.records]
        assert any("roc_auc" in m and "0.85" in m for m in messages)
        assert any("ks" in m and "0.42" in m for m in messages)

    def test_never_raises_on_empty_dict(self):
        log_metrics({})  # must not raise


class TestLogParams:
    def test_scalars_logged_as_is(self, caplog):
        with caplog.at_level(logging.INFO, logger="dscompanion.tracking.run_context"):
            log_params({"lr": 0.01, "name": "xgb", "flag": True})
        assert len(caplog.records) == 1
        message = caplog.records[0].message
        assert "0.01" in message and "xgb" in message and "True" in message

    def test_non_scalar_coerced_to_str(self, caplog):
        with caplog.at_level(logging.INFO, logger="dscompanion.tracking.run_context"):
            log_params({"cols": ["a", "b"], "cfg": {"k": 1}})
        message = caplog.records[0].message
        assert "['a', 'b']" in message
        assert "{'k': 1}" in message


class TestLogArtifact:
    def test_logs_path_and_destination(self, caplog):
        with caplog.at_level(logging.INFO, logger="dscompanion.tracking.run_context"):
            log_artifact("/tmp/model.pkl", artifact_path="models/")
        assert len(caplog.records) == 1
        message = caplog.records[0].message
        assert "/tmp/model.pkl" in message and "models/" in message

    def test_logs_root_when_no_artifact_path(self, caplog):
        with caplog.at_level(logging.INFO, logger="dscompanion.tracking.run_context"):
            log_artifact("/tmp/model.pkl")
        assert len(caplog.records) == 1
