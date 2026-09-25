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
    assert "--review" in steps[3][1] and "--tenth-sweep" in steps[-1][1]  # sweep reads the review tables


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
    codes = iter([1, 0, 0, 0, 0, 0])  # niftyindices.com blocked, everything else fine
    with patch.object(refresh_data, "MANIFEST_PATH", tmp_path / "m.json"), \
            patch("refresh_data.subprocess.Popen", side_effect=lambda *a, **k: _Proc(next(codes))):
        result = run_refresh(date(2026, 10, 3), on_output=lambda line: None)
    assert result["status"] == "ok" and len(result["steps"]) == 6
    assert len(result["warnings"]) == 1 and refresh_data.TRI_STEP in result["warnings"][0]
