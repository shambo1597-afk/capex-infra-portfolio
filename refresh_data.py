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
Steps 1-5 are required: the refresh stops at the first failure among them.

Every step uses the same end date, so all files describe the same price window. The outcome is
written to output/refresh_manifest.json, which the dashboard shows (including a failed step,
since the files may then come from different runs). Needs internet access to NSE and Screener.in.
"""

import json
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, List, Optional, Tuple
from zoneinfo import ZoneInfo

from config import BASE_DIR, OUTPUT_DIR

MANIFEST_PATH = OUTPUT_DIR / "refresh_manifest.json"
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
    ]


def run_refresh(as_of: Optional[date] = None, on_output: Callable[[str], None] = print) -> dict:
    """Run all steps; return (and save) the manifest. on_output receives each output line."""
    as_of = as_of or datetime.now(IST).date()
    manifest = {"as_of": as_of.isoformat(), "started": datetime.now(IST).isoformat(timespec="seconds"),
                "steps": [], "status": "running"}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")  # marks a refresh in progress
    for name, cmd in refresh_steps(as_of):
        on_output(f"=== {name} ===")
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        tail: List[str] = []
        for line in proc.stdout:
            line = line.rstrip()
            on_output(line)
            tail = (tail + [line])[-15:]
        proc.wait()
        step = {"name": name, "returncode": proc.returncode, "seconds": round(time.time() - t0)}
        manifest["steps"].append(step)
        if proc.returncode != 0 and name == TRI_STEP:
            manifest.setdefault("warnings", []).append(
                f"{name} failed; the dashboard keeps the existing TRI file. Last output: {tail[-1] if tail else ''}")
            continue
        if proc.returncode != 0:
            manifest.update(status="failed", failed_step=name, error_tail=tail)
            break
    else:
        manifest["status"] = "ok"
    manifest["finished"] = datetime.now(IST).isoformat(timespec="seconds")
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


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
