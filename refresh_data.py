"""
Refresh every output the dashboard reads, in dependency order, for one pinned end date.

    python refresh_data.py                 # end date = today (India time)
    python refresh_data.py --as-of 2026-10-03

Steps (each a separate process):
  0. fetch_data.py --update-tri       Nifty 500 TRI from niftyindices.com appended to the local CSV
                                      (optional: bot protection can block it; a failure is recorded
                                      as a warning and the refresh continues with the existing CSV)
  1. main.py                          prices (NSE Bhavcopy), technical summary, risk summary
  2. sector_screen.py                 technical-first screens (live fundamentals for passers)
  3. sector_screen.py --review        review tables (live fundamentals, RRG) and RRG plots
  4. sector_screen.py --locked-check  run-up / results-date check of the locked stocks
  5. sector_screen.py --tenth-sweep   candidate sweep over the rest of the universe
  6. rrg_tails.py                     weekly RRG tails: holdings and sector rotation (NSE sector indices)
  7. risk_model.py                    factor series, regressions, Nifty F&O prices and the hedge plan
  8. tracker.py                       live value and P&L of the Rs 1 crore from the 28-Sep snapshot
  9. performance.py                   Sharpe, Treynor, XIRR (backtest, and live from the tracker), CML
Steps 1-9 are required: the refresh stops at the first failure among them.

Every step uses the same end date, so all files describe the same price window. Progress and the
outcome are written to output/refresh_manifest.json (current step, a heartbeat every few seconds,
per-step durations, any failure), which the dashboard polls. The dashboard starts the refresh as
a separate background process (start_background_refresh), so closing or reloading the page does
not interrupt it; its output goes to output/refresh_log.txt. Needs internet access to NSE,
Screener.in and niftyindices.com.
"""

import json
import os
import subprocess
import sys
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, List, Optional, Tuple
from zoneinfo import ZoneInfo

from config import BASE_DIR, OUTPUT_DIR

MANIFEST_PATH = OUTPUT_DIR / "refresh_manifest.json"
LOG_PATH = OUTPUT_DIR / "refresh_log.txt"
HEARTBEAT_SECONDS = 5
# A running refresh whose heartbeat is older than this was interrupted (process killed, machine
# slept, app restarted) and will never finish; the dashboard then offers to start a new one
STALLED_AFTER_SECONDS = 60
IST = ZoneInfo("Asia/Kolkata")


TRI_STEP = "Nifty 500 TRI (niftyindices.com)"


def refresh_steps(as_of: date) -> List[Tuple[str, List[str]]]:
    end = as_of.isoformat()
    start = (as_of - timedelta(days=365)).isoformat()
    py = sys.executable
    return [
        (TRI_STEP, [py, "fetch_data.py", "--update-tri", end]),
        ("Prices, technicals and risk summary", [py, "main.py", "--start-date", start, "--end-date", end]),
        ("Technical-first sector screens", [py, "sector_screen.py", "--as-of", end]),
        ("Review tables, RRG and plots", [py, "sector_screen.py", "--review", "--as-of", end]),
        ("Locked-portfolio run-up check", [py, "sector_screen.py", "--locked-check", "--as-of", end]),
        ("Candidate sweep", [py, "sector_screen.py", "--tenth-sweep", "--as-of", end]),
        ("RRG weekly tails", [py, "rrg_tails.py", "--as-of", end]),
        ("Regression and hedge plan", [py, "risk_model.py", "--as-of", end]),
        ("Live P&L of the Rs 1 crore", [py, "tracker.py", "--as-of", end]),
        ("Performance and CML", [py, "performance.py", "--as-of", end]),
    ]


class _Manifest:
    """The progress/outcome record, written atomically (the dashboard may read it mid-write)."""

    def __init__(self, data: dict, path: Path):
        self.data, self.path, self.lock = data, path, threading.Lock()

    def update(self, **fields) -> None:
        with self.lock:
            self.data.update(fields)
            self.data["heartbeat"] = datetime.now(IST).isoformat(timespec="seconds")
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
            tmp.replace(self.path)


def run_refresh(as_of: Optional[date] = None, on_output: Callable[[str], None] = print) -> dict:
    """Run all steps; return (and save) the manifest. on_output receives each output line."""
    as_of = as_of or datetime.now(IST).date()
    steps = refresh_steps(as_of)
    expected = _expected_seconds(load_manifest(MANIFEST_PATH))
    manifest = _Manifest({"as_of": as_of.isoformat(), "started": datetime.now(IST).isoformat(timespec="seconds"),
                          "status": "running", "pid": os.getpid(), "total_steps": len(steps),
                          "step_names": [name for name, _ in steps], "expected_seconds": expected,
                          "steps": []}, MANIFEST_PATH)
    manifest.update()
    done = threading.Event()

    def heartbeat() -> None:
        while not done.wait(HEARTBEAT_SECONDS):
            manifest.update()

    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        for index, (name, cmd) in enumerate(steps, start=1):
            manifest.update(current_step=index, current_step_name=name,
                            current_step_started=datetime.now(IST).isoformat(timespec="seconds"))
            on_output(f"=== Step {index}/{len(steps)}: {name} ===")
            t0 = time.time()
            proc = subprocess.Popen(cmd, cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            tail: List[str] = []
            for line in proc.stdout:
                line = line.rstrip()
                on_output(line)
                tail = (tail + [line])[-15:]
            proc.wait()
            manifest.data["steps"].append({"name": name, "returncode": proc.returncode,
                                           "seconds": round(time.time() - t0)})
            if proc.returncode != 0 and name == TRI_STEP:
                manifest.data.setdefault("warnings", []).append(
                    f"{name} failed; the dashboard keeps the existing TRI file. Last output: {tail[-1] if tail else ''}")
                manifest.update()
                continue
            if proc.returncode != 0:
                manifest.data.update(status="failed", failed_step=name, error_tail=tail)
                break
        else:
            manifest.data["status"] = "ok"
    except BaseException as exc:  # interrupted (Ctrl+C, killed): never leave "running" behind
        manifest.data.update(status="failed", failed_step=manifest.data.get("current_step_name"),
                             error_tail=[f"{type(exc).__name__}: {exc}"])
        raise
    finally:
        done.set()
        manifest.update(finished=datetime.now(IST).isoformat(timespec="seconds"))
    return manifest.data


def _expected_seconds(previous: Optional[dict]) -> dict:
    """Per-step durations of the last successful refresh (for the time-remaining estimate)."""
    previous = previous or {}
    if previous.get("status") == "ok" and previous.get("steps"):
        return {st["name"]: st["seconds"] for st in previous["steps"]}
    return previous.get("expected_seconds", {})


def refresh_state(manifest: Optional[dict], now: Optional[datetime] = None) -> Optional[str]:
    """
    "running", "stalled" (running but no heartbeat for STALLED_AFTER_SECONDS: interrupted),
    "ok", "failed", or None when no refresh has been recorded.
    """
    if not manifest:
        return None
    status = manifest.get("status")
    if status != "running":
        return status
    now = now or datetime.now(IST)
    beat = manifest.get("heartbeat") or manifest.get("started")
    age = (now - datetime.fromisoformat(beat)).total_seconds()
    return "running" if age <= STALLED_AFTER_SECONDS else "stalled"


def start_background_refresh() -> bool:
    """
    Start `python refresh_data.py` as a detached process (it survives the dashboard page being
    closed or reloaded), logging to LOG_PATH. Returns False if a refresh is already running.
    """
    if refresh_state(load_manifest()) == "running":
        return False
    # Mark it as running first, so the page shows progress while the process starts up
    now = datetime.now(IST).isoformat(timespec="seconds")
    MANIFEST_PATH.write_text(json.dumps({
        "status": "running", "started": now, "heartbeat": now, "steps": [],
        "total_steps": len(refresh_steps(datetime.now(IST).date())),
        "expected_seconds": _expected_seconds(load_manifest())}, indent=2), encoding="utf-8")
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    with open(LOG_PATH, "w", encoding="utf-8") as log:
        subprocess.Popen([sys.executable, "-u", "refresh_data.py"], cwd=BASE_DIR, stdout=log,
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kwargs)
    return True


def progress_summary(manifest: dict, now: Optional[datetime] = None) -> dict:
    """
    Progress of a running refresh for the dashboard: step (1-based) and total, step name, seconds
    in the current step, overall fraction done, and estimated seconds left. The estimate uses the
    last successful refresh's step durations; without one, remaining_seconds is None and the
    fraction counts completed steps.
    """
    now = now or datetime.now(IST)
    total = manifest.get("total_steps") or len(manifest.get("step_names", [])) or 1
    names = manifest.get("step_names") or []
    done_steps = manifest.get("steps", [])
    step = manifest.get("current_step") or (len(done_steps) + 1)
    started = manifest.get("current_step_started")
    elapsed = (now - datetime.fromisoformat(started)).total_seconds() if started else 0.0
    expected = manifest.get("expected_seconds") or {}
    remaining = None
    overrunning = False
    fraction = min(len(done_steps) / total, 1.0)
    if names and all(n in expected for n in names):
        total_expected = sum(expected[n] for n in names) or 1
        finished = sum(expected[st["name"]] for st in done_steps if st["name"] in expected)
        current = names[step - 1] if 0 < step <= len(names) and len(done_steps) < step else None
        in_step = min(elapsed, expected[current]) if current else 0
        fraction = min((finished + in_step) / total_expected, 0.99)
        later = sum(expected[n] for n in names[step:]) if current else 0
        remaining = max((expected[current] - elapsed) if current else 0, 0) + later
        overrunning = bool(current) and elapsed > expected[current] + 15
    return {"step": min(step, total), "total": total, "step_name": manifest.get("current_step_name", "Starting"),
            "step_elapsed": elapsed, "fraction": fraction, "remaining_seconds": remaining,
            "overrunning": overrunning}


def read_log_tail(lines: int = 20) -> List[str]:
    try:
        return LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return []


def latest_expected_session(now: Optional[datetime] = None) -> date:
    """
    The most recent session whose NSE Bhavcopy should be published by `now` (India time): today
    after 19:00 IST on a weekday, otherwise the previous weekday. Exchange holidays are not
    modelled, so on the day after a holiday data can look one session stale when it is not.
    """
    now = now.astimezone(IST) if now else datetime.now(IST)
    day = now.date() if now.hour >= 19 else now.date() - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def load_manifest(path: Path = MANIFEST_PATH) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


if __name__ == "__main__":
    as_of_arg = None
    if "--as-of" in sys.argv:
        as_of_arg = date.fromisoformat(sys.argv[sys.argv.index("--as-of") + 1])
    result = run_refresh(as_of_arg)
    print(f"\nRefresh {result['status']}" + (f" at step: {result.get('failed_step')}" if result["status"] != "ok" else ""))
    sys.exit(0 if result["status"] == "ok" else 1)
