"""
Tests for the one-click data refresh (refresh_data.py) that the dashboard's button runs.
"""

from datetime import date, datetime
from unittest.mock import patch

import pytest

import refresh_data
from refresh_data import IST, latest_expected_session, refresh_steps, run_refresh


@pytest.mark.parametrize("now, expected", [
    (datetime(2026, 9, 25, 20, 0, tzinfo=IST), date(2026, 9, 25)),   # Friday evening: today's file is out
    (datetime(2026, 9, 25, 14, 0, tzinfo=IST), date(2026, 9, 24)),   # Friday afternoon: yesterday's
    (datetime(2026, 9, 26, 12, 0, tzinfo=IST), date(2026, 9, 25)),   # Saturday: Friday
    (datetime(2026, 9, 28, 9, 0, tzinfo=IST), date(2026, 9, 25)),    # Monday morning: Friday
])
def test_latest_expected_session(now, expected):
    assert latest_expected_session(now) == expected


def test_every_step_uses_the_same_end_date():
    steps = refresh_steps(date(2026, 10, 3))
    names = [name for name, _ in steps]
    assert names[0] == refresh_data.TRI_STEP and names[1].startswith("Prices")  # prices before their readers
    for _, cmd in steps:
        assert "2026-10-03" in cmd
    assert "--review" in steps[3][1] and "--tenth-sweep" in steps[5][1]  # sweep reads the review tables
    # the regressions read the risk summary; performance reads the factor series they write
    assert [s[1][1] for s in steps[-3:]] == ["risk_model.py", "performance.py", "tracker.py"]


class _Proc:
    def __init__(self, code):
        self.stdout, self.returncode = iter(["line 1\n", "line 2\n"]), code

    def wait(self):
        return self.returncode


def test_failed_step_stops_the_refresh_and_is_recorded(tmp_path):
    codes = iter([0, 0, 1])  # TRI ok, prices ok, screens fail
    with patch.object(refresh_data, "MANIFEST_PATH", tmp_path / "m.json"), \
            patch("refresh_data.subprocess.Popen", side_effect=lambda *a, **k: _Proc(next(codes))):
        result = run_refresh(date(2026, 10, 3), on_output=lambda line: None)
    assert result["status"] == "failed" and len(result["steps"]) == 3
    assert result["failed_step"] == "Technical-first sector screens" and result["error_tail"] == ["line 1", "line 2"]
    assert refresh_data.load_manifest(tmp_path / "m.json")["status"] == "failed"


def test_successful_refresh(tmp_path):
    with patch.object(refresh_data, "MANIFEST_PATH", tmp_path / "m.json"), \
            patch("refresh_data.subprocess.Popen", side_effect=lambda *a, **k: _Proc(0)):
        result = run_refresh(date(2026, 10, 3), on_output=lambda line: None)
    assert result["status"] == "ok" and len(result["steps"]) == len(refresh_steps(date(2026, 10, 3)))


def test_tri_failure_is_a_warning_not_a_failed_refresh(tmp_path):
    codes = iter([1] + [0] * 9)  # niftyindices.com blocked, everything else fine
    with patch.object(refresh_data, "MANIFEST_PATH", tmp_path / "m.json"), \
            patch("refresh_data.subprocess.Popen", side_effect=lambda *a, **k: _Proc(next(codes))):
        result = run_refresh(date(2026, 10, 3), on_output=lambda line: None)
    assert result["status"] == "ok" and len(result["steps"]) == len(refresh_steps(date(2026, 10, 3)))
    assert len(result["warnings"]) == 1 and refresh_data.TRI_STEP in result["warnings"][0]


NOW = datetime(2026, 9, 25, 20, 40, 0, tzinfo=IST)


class TestRefreshState:
    def test_running_with_recent_heartbeat(self):
        m = {"status": "running", "heartbeat": "2026-09-25T20:39:30+05:30"}
        assert refresh_data.refresh_state(m, NOW) == "running"

    def test_running_without_heartbeat_for_over_a_minute_is_stalled(self):
        m = {"status": "running", "heartbeat": "2026-09-25T20:34:00+05:30"}
        assert refresh_data.refresh_state(m, NOW) == "stalled"

    def test_finished_states_and_none(self):
        assert refresh_data.refresh_state({"status": "ok"}, NOW) == "ok"
        assert refresh_data.refresh_state({"status": "failed"}, NOW) == "failed"
        assert refresh_data.refresh_state(None, NOW) is None


class TestProgressSummary:
    NAMES = ["A", "B", "C"]

    def test_estimate_from_last_refresh_durations(self):
        m = {"total_steps": 3, "step_names": self.NAMES, "expected_seconds": {"A": 20, "B": 60, "C": 20},
             "steps": [{"name": "A", "returncode": 0, "seconds": 18}], "current_step": 2,
             "current_step_name": "B", "current_step_started": "2026-09-25T20:39:30+05:30"}
        p = refresh_data.progress_summary(m, NOW)
        assert (p["step"], p["total"], p["step_name"], p["step_elapsed"]) == (2, 3, "B", 30)
        assert p["fraction"] == pytest.approx((20 + 30) / 100)
        assert p["remaining_seconds"] == pytest.approx(30 + 20) and not p["overrunning"]

    def test_overrunning_step_never_shows_negative_time_or_100_percent(self):
        m = {"total_steps": 3, "step_names": self.NAMES, "expected_seconds": {"A": 20, "B": 10, "C": 20},
             "steps": [{"name": "A", "returncode": 0, "seconds": 18}], "current_step": 2,
             "current_step_name": "B", "current_step_started": "2026-09-25T20:30:00+05:30"}
        p = refresh_data.progress_summary(m, NOW)
        assert p["remaining_seconds"] == 20 and p["fraction"] < 1 and p["overrunning"]

    def test_without_history_counts_completed_steps(self):
        m = {"total_steps": 4, "step_names": ["A", "B", "C", "D"], "expected_seconds": {},
             "steps": [{"name": "A", "returncode": 0, "seconds": 5}], "current_step": 2, "current_step_name": "B",
             "current_step_started": "2026-09-25T20:39:55+05:30"}
        p = refresh_data.progress_summary(m, NOW)
        assert p["fraction"] == 0.25 and p["remaining_seconds"] is None


def test_manifest_records_progress_and_next_estimate(tmp_path):
    path = tmp_path / "m.json"
    seen = []

    def popen(*args, **kwargs):
        seen.append(refresh_data.load_manifest(path))  # what the dashboard would read mid-step
        return _Proc(0)
    with patch.object(refresh_data, "MANIFEST_PATH", path), patch("refresh_data.subprocess.Popen", side_effect=popen):
        result = run_refresh(date(2026, 10, 3), on_output=lambda line: None)
    assert [m["current_step"] for m in seen] == list(range(1, len(refresh_steps(date(2026, 10, 3))) + 1)) and all("heartbeat" in m for m in seen)
    assert seen[0]["status"] == "running" and seen[0]["total_steps"] == len(refresh_steps(date(2026, 10, 3)))
    assert result["status"] == "ok" and "finished" in result
    # The next refresh estimates its time from this one's step durations
    assert refresh_data._expected_seconds(result) == {s["name"]: s["seconds"] for s in result["steps"]}


def test_interrupted_refresh_is_marked_failed_not_left_running(tmp_path):
    path = tmp_path / "m.json"

    def popen(*args, **kwargs):
        raise KeyboardInterrupt
    with patch.object(refresh_data, "MANIFEST_PATH", path), patch("refresh_data.subprocess.Popen", side_effect=popen):
        with pytest.raises(KeyboardInterrupt):
            run_refresh(date(2026, 10, 3), on_output=lambda line: None)
    saved = refresh_data.load_manifest(path)
    assert saved["status"] == "failed" and saved["failed_step"] == refresh_data.TRI_STEP and "finished" in saved
