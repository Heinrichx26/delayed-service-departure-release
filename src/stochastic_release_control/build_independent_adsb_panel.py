"""Build and evaluate an independent ADS-B departure-calendar replication."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from math import ceil
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
from .joint_replication.extensions import PairGreedy, old
from .joint_replication.joint_design import JointDesign
from .joint_replication.validate_joint import replay

SOURCE = ROOT / "data" / "independent_adsb" / "processed" / "airport_departures_2025_jan.csv"
AIRPORTS = {"ATL": "America/New_York", "CLT": "America/New_York", "DFW": "America/Chicago"}
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


def load_panel(airport_code: str) -> pd.DataFrame:
    frame = pd.read_csv(SOURCE)
    frame = frame[frame["airport"] == airport_code].copy()
    frame["departure_utc"] = pd.to_datetime(frame["departure_utc"], utc=True)
    frame["departure_local"] = frame["departure_utc"].dt.tz_convert(ZoneInfo(AIRPORTS[airport_code]))
    frame["local_date"] = frame["departure_local"].dt.date
    frame["local_minute"] = (
        frame["departure_local"].dt.hour * 60
        + frame["departure_local"].dt.minute
        + frame["departure_local"].dt.second / 60.0
    )
    return frame


def profile(frame: pd.DataFrame, start_minute: int) -> list[int]:
    raw = np.zeros(HORIZON - LAG, dtype=int)
    relative = frame["local_minute"].to_numpy() - start_minute
    for value in relative:
        slot = int(np.floor(value / 2.0))
        if 0 <= slot < HORIZON - LAG:
            raw[slot] += 1
    return [0] * LAG + raw.tolist()


def make_case(frame: pd.DataFrame, airport_code: str, day: int, hour: int) -> dict | None:
    donors = [
        profile(frame[frame["local_date"] == pd.Timestamp(f"2025-01-{donor:02d}").date()], hour * 60)
        for donor in DONOR_DAYS
    ]
    evaluation = profile(
        frame[frame["local_date"] == pd.Timestamp(f"2025-01-{day:02d}").date()],
        hour * 60,
    )
    n = min(sum(path) for path in donors + [evaluation])
    if n < 2:
        return None
    key = f"{airport_code}_adsb_2025-01-{day:02d}_h{hour:02d}"
    return {
        "key": key,
        "airport": airport_code,
        "day": f"2025-01-{day:02d}",
        "hour": hour,
        "n": int(n),
        "G": int(2 * n),
        "lag": LAG,
        "delta": 2,
        "costs": [2, 3, 0],
        "service": donors,
        "probabilities": ["1/16"] * 16,
        "donor_days": [f"2025-01-{d:02d}" for d in DONOR_DAYS],
        "caps": [int(n)] * HORIZON,
        "actual_service": evaluation,
        "data_source": "data/independent_adsb/processed/airport_departures_2025_jan.csv",
        "service_definition": "ADS-B-derived departure times converted to local two-minute service slots after a six-slot transport lag.",
        "selection_rule": "Minimum donor/evaluation service total at the airport-hour window; retain n>=2.",
    }


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
    replay(problem, exact.replay(), lead, delay, budget)
    greedy = PairGreedy(problem, lead, delay, budget)
    greedy_answer = greedy.solve()
    replay(problem, greedy.replay(), lead, delay, budget)
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


def build(phase: str, airport_code: str) -> dict:
    data_root = ROOT / "data" / "independent_adsb" / "cases" / airport_code
    results_root = ROOT / "results" / "independent_adsb_replication" / airport_code
    frame = load_panel(airport_code)
    cases = []
    for day in EVAL_DAYS:
        for hour in START_HOURS:
            case = make_case(frame, airport_code, day, hour)
            if case is None:
                continue
            case_id = hashlib.sha256(case["key"].encode()).hexdigest()[:12]
            case["case_id"] = case_id
            path = data_root / f"{case_id}.json"
            save(path, case)
            cases.append({
                "case_id": case_id,
                "input_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "airport": airport_code,
                "day": case["day"],
                "hour": hour,
                "capacity": "N",
                "divisor": 1,
            })
    save(
        data_root.parent / "panel_design.json",
        {
            "airport": airport_code,
            "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
            "source_sha256": digest(SOURCE),
            "timezone": AIRPORTS[airport_code],
            "donor_days": list(DONOR_DAYS),
            "evaluation_days": list(EVAL_DAYS),
            "start_hours": list(START_HOURS),
            "cases": cases,
        },
    )
    smoke_case = dict(cases[0], capacity="N4", divisor=4)
    smoke_result = evaluate_one((smoke_case, 1, 1))
    smoke = {"passed": bool(smoke_result["exact_optimal"] and smoke_result["replay_checked"]), "row": smoke_result}
    save(results_root / "smoke.json", smoke)
    if phase == "smoke":
        print(json.dumps(smoke, indent=2))
        return smoke
    if not smoke["passed"]:
        raise RuntimeError("Independent ADS-B smoke test failed")
    jobs = [dict(case, capacity=label, divisor=divisor) for case in cases for label, divisor in (("N", 1), ("N4", 4), ("N8", 8))]
    tasks = [(case, delay, budget) for case in jobs for delay in (1, 2) for budget in (1, 2)]
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
                "replay_checked": all(r["replay_checked"] for r in group),
            })
    save(results_root / "summary.json", summary)
    save(results_root / "audit.json", {"passed": all(r["replay_checked"] and r["exact_optimal"] for r in rows), "conditions": len(rows), "source_sha256": digest(SOURCE)})
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["smoke", "full"])
    parser.add_argument("--airport", required=True, choices=sorted(AIRPORTS))
    args = parser.parse_args()
    build(args.phase, args.airport)
