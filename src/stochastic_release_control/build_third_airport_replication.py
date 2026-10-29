"""Build and evaluate a third-airport replication from public BTS records.

The replication keeps the joint information model unchanged.  It uses the
same two-minute decision slots, the same five-minute service counts converted
to two-minute bins, the same sixteen January 3--18 donor dates, and the same
response/request settings used for the DEN--PHX panel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from math import ceil
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
from .joint_replication.extensions import PairGreedy, old
from .joint_replication.joint_design import JointDesign
from .joint_replication.validate_joint import replay

SOURCE = ROOT / "data" / "third_airport_replication" / "bts_calendar_corrected.csv.gz"
HORIZON = 36
LAG = 6
START_HOURS = (10, 12, 14, 16, 18)
EVAL_DAYS = tuple(range(24, 32))
DONOR_DAYS = tuple(range(3, 19))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def profile(frame: pd.DataFrame, start_minute: int) -> list[int]:
    """Count recorded takeoffs in two-minute bins after the fixed six-slot lag."""
    raw = np.zeros(HORIZON - LAG, dtype=int)
    relative = frame["takeoff_minute"].to_numpy() - start_minute
    for value in relative:
        slot = int(np.floor(value / 2.0))
        if 0 <= slot < HORIZON - LAG:
            raw[slot] += 1
    return [0] * LAG + raw.tolist()


def load_panel() -> pd.DataFrame:
    frame = pd.read_csv(SOURCE, compression="gzip")
    frame["operation_date"] = pd.to_datetime(frame["operation_date"])
    frame["actual_takeoff"] = pd.to_datetime(frame["actual_takeoff"])
    frame["takeoff_minute"] = (
        frame["actual_takeoff"] - frame["operation_date"]
    ).dt.total_seconds() / 60.0
    return frame


def make_case(frame: pd.DataFrame, airport_code: str, year: int, day: int, hour: int) -> dict | None:
    airport = frame[(frame["airport"] == airport_code) & (frame["source_year"] == year)]
    donors = []
    for donor in DONOR_DAYS:
        date = pd.Timestamp(f"{year}-01-{donor:02d}")
        donors.append(profile(airport[airport["operation_date"] == date], hour * 60))
    evaluation = profile(
        airport[airport["operation_date"] == pd.Timestamp(f"{year}-01-{day:02d}")],
        hour * 60,
    )
    n = min(sum(path) for path in donors + [evaluation])
    if n < 2:
        return None
    key = f"{airport_code}_{year:04d}-01-{day:02d}_h{hour:02d}"
    return {
        "key": key,
        "airport": airport_code,
        "day": f"{year:04d}-01-{day:02d}",
        "hour": hour,
        "n": int(n),
        "G": int(2 * n),
        "lag": LAG,
        "delta": 2,
        "costs": [2, 3, 0],
        "service": donors,
        "probabilities": ["1/16"] * 16,
        "donor_days": [f"{year:04d}-01-{d:02d}" for d in DONOR_DAYS],
        "caps": [int(n)] * HORIZON,
        "actual_service": evaluation,
        "data_source": "data/third_airport_replication/bts_calendar_corrected.csv.gz",
        "service_definition": "Recorded WheelsOff takeoffs counted in two-minute slots after a six-slot transport lag.",
        "selection_rule": "Minimum donor/evaluation service total at the airport-hour window; retain n>=2.",
    }


def build(phase: str, airport_code: str) -> dict:
    if airport_code == "ATL":
        data_root = ROOT / "data" / "third_airport_replication"
        results_root = ROOT / "results" / "third_airport_replication"
    else:
        data_root = ROOT / "data" / "third_airport_replication" / airport_code
        results_root = ROOT / "results" / "external_airport_replication" / airport_code
    frame = load_panel()
    cases = []
    for year in (2024, 2025):
        for day in EVAL_DAYS:
            for hour in START_HOURS:
                case = make_case(frame, airport_code, year, day, hour)
                if case is None:
                    continue
                case_id = hashlib.sha256(case["key"].encode()).hexdigest()[:12]
                case["case_id"] = case_id
                path = data_root / "cases" / f"{case_id}.json"
                save(path, case)
                cases.append({
                    "case_id": case_id,
                    "input_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "input_sha256": digest(path),
                    "airport": airport_code,
                    "day": case["day"],
                    "hour": case["hour"],
                    "capacity": "N",
                    "divisor": 1,
                })
    manifest = {
        "date": "2026-10-28",
        "airport": airport_code,
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": digest(SOURCE),
        "selection": "January 3--18 donor dates and January 24--31 evaluation dates in 2024 and 2025; five starts at 10:00--18:00 local time; retain n>=2.",
        "slot_minutes": 2,
        "horizon_slots": HORIZON,
        "lag_slots": LAG,
        "cases": cases,
        "delays": [1, 2],
        "budgets": [1, 2],
    }
    save(data_root / "panel_design.json", manifest)
    smoke_case = dict(cases[0], capacity="N4", divisor=4)
    smoke_result = evaluate_one((smoke_case, 1, 1))
    smoke = {"passed": bool(smoke_result["exact_optimal"] and smoke_result["replay_checked"]), "row": smoke_result}
    save(results_root / "smoke.json", smoke)
    if phase == "smoke":
        print(json.dumps(smoke, indent=2))
        return smoke
    if not smoke["passed"]:
        raise RuntimeError("Third-airport smoke test failed")
    capacity_jobs = []
    for case in cases:
        for label, divisor in (("N", 1), ("N4", 4), ("N8", 8)):
            capacity_jobs.append(dict(case, capacity=label, divisor=divisor))
    tasks = [(case, delay, budget) for case in capacity_jobs for delay in (1, 2) for budget in (1, 2)]
    rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for row in pool.map(evaluate_one, tasks):
            rows.append(row)
    save(results_root / "rows.json", rows)
    summary = []
    for delay in (1, 2):
        for budget in (1, 2):
            group = [r for r in rows if r["delay"] == delay and r["budget"] == budget]
            summary.append({
                "airport": airport_code,
                "delay": delay,
                "budget": budget,
                "conditions": len(group),
                "exact_waiting": mean(r["exact_waiting"] for r in group),
                "greedy_waiting": mean(r["greedy_waiting"] for r in group),
                "saving": mean(r["saving"] for r in group),
                "strictly_improved": sum(r["saving"] > 1e-9 for r in group),
                "greedy_requests": mean(r["greedy_requests"] for r in group),
                "exact_requests": mean(r["exact_requests"] for r in group),
                "replay_checked": all(r["replay_checked"] for r in group),
            })
    save(results_root / "summary.json", summary)
    save(results_root / "audit.json", {"passed": all(r["replay_checked"] and r["exact_optimal"] for r in rows), "conditions": len(rows), "source_sha256": digest(SOURCE)})
    print(json.dumps(summary, indent=2))
    return summary


def evaluate_one(task: tuple[dict, int, int]) -> dict:
    job, delay, budget = task
    case = json.loads((ROOT / job["input_path"]).read_text(encoding="utf-8"))
    cap = ceil(case["n"] / job["divisor"])
    case["caps"] = [cap] * len(case["caps"])
    problem = old.prepare(case)
    if problem is None:
        raise RuntimeError(f"incomplete support: {job['case_id']}")
    lead = case["lag"] + 2
    exact = JointDesign(problem, lead, delay, budget)
    exact_answer = exact.solve()
    exact_paths = exact.replay()
    replay(problem, exact_paths, lead, delay, budget)
    greedy = PairGreedy(problem, lead, delay, budget)
    greedy_answer = greedy.solve()
    greedy_paths = greedy.replay()
    replay(problem, greedy_paths, lead, delay, budget)
    return {
        "case_id": job["case_id"],
        "airport": job["airport"],
        "day": job["day"],
        "hour": job["hour"],
        "n": case["n"],
        "capacity": job["capacity"],
        "delay": delay,
        "budget": budget,
        "exact_waiting": exact_answer["waiting"],
        "greedy_waiting": greedy_answer["waiting"],
        "saving": greedy_answer["waiting"] - exact_answer["waiting"],
        "exact_requests": exact_answer["expected_queries"],
        "greedy_requests": greedy_answer["expected_queries"],
        "exact_states": exact_answer["states"],
        "greedy_states": greedy_answer.get("states"),
        "exact_optimal": bool(exact_answer["optimal"]),
        "replay_checked": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["smoke", "full"])
    parser.add_argument("--airport", required=True, choices=["ATL", "CLT", "DFW"])
    args = parser.parse_args()
    build(args.phase, args.airport)
