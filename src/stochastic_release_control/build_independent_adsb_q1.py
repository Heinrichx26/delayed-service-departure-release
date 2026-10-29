"""Build and evaluate a quarterly independent ADS-B replication."""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from math import ceil
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_SOURCE = ROOT / "data" / "independent_adsb" / "raw" / "2025_Q1_detailed_github.parquet"
PROCESSED_SOURCE = ROOT / "data" / "independent_adsb_q1" / "processed" / "airport_departures_2025_q1.csv"
AIRPORTS = {"ATL": ("KATL", "America/New_York"), "CLT": ("KCLT", "America/New_York"), "DFW": ("KDFW", "America/Chicago")}
MONTH_EVAL_DAYS = {1: tuple(range(24, 32)), 2: tuple(range(24, 29)), 3: tuple(range(24, 32))}
DONOR_DAYS = tuple(range(3, 19))
START_HOURS = (10, 12, 14, 16, 18)
HORIZON = 36
LAG = 6

from .joint_replication.extensions import PairGreedy, old
from .joint_replication.joint_design import JointDesign
from .joint_replication.validate_joint import replay


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare_processed() -> None:
    if PROCESSED_SOURCE.exists():
        return
    cols = ["ICAO_Hex", "Callsign", "Track_Origin_DateTime_UTC", "Track_Origin_ApplicableAirports"]
    frame = pd.read_parquet(RAW_SOURCE, columns=cols)
    airport_codes = {code: f"['{icao}']" for code, (icao, _) in AIRPORTS.items()}
    frame = frame[frame["Track_Origin_ApplicableAirports"].isin(airport_codes.values())].copy()
    frame["departure_utc"] = pd.to_datetime(frame["Track_Origin_DateTime_UTC"], errors="coerce", utc=True)
    frame = frame[frame["departure_utc"].notna()].copy()
    frame = frame[(frame["departure_utc"] >= "2025-01-01") & (frame["departure_utc"] < "2025-04-01")]
    inverse = {value: key for key, value in airport_codes.items()}
    frame["airport"] = frame["Track_Origin_ApplicableAirports"].map(inverse)
    frame = frame[["airport", "departure_utc", "Callsign", "ICAO_Hex"]].sort_values("departure_utc")
    PROCESSED_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PROCESSED_SOURCE, index=False, date_format="%Y-%m-%d %H:%M:%S")


def load_panel(airport: str) -> pd.DataFrame:
    frame = pd.read_csv(PROCESSED_SOURCE)
    frame = frame[frame["airport"] == airport].copy()
    frame["departure_utc"] = pd.to_datetime(frame["departure_utc"], utc=True)
    frame["departure_local"] = frame["departure_utc"].dt.tz_convert(ZoneInfo(AIRPORTS[airport][1]))
    frame["local_date"] = frame["departure_local"].dt.date
    frame["local_minute"] = frame["departure_local"].dt.hour * 60 + frame["departure_local"].dt.minute + frame["departure_local"].dt.second / 60.0
    return frame


def profile(frame: pd.DataFrame, start_minute: int) -> list[int]:
    raw = np.zeros(HORIZON - LAG, dtype=int)
    relative = frame["local_minute"].to_numpy() - start_minute
    for value in relative:
        slot = int(np.floor(value / 2.0))
        if 0 <= slot < HORIZON - LAG:
            raw[slot] += 1
    return [0] * LAG + raw.tolist()


def make_case(frame: pd.DataFrame, airport: str, month: int, day: int, hour: int) -> dict | None:
    year_month = f"2025-{month:02d}"
    donors = [profile(frame[frame["local_date"] == pd.Timestamp(f"{year_month}-{d:02d}").date()], hour * 60) for d in DONOR_DAYS]
    evaluation = profile(frame[frame["local_date"] == pd.Timestamp(f"{year_month}-{day:02d}").date()], hour * 60)
    n = min(sum(path) for path in donors + [evaluation])
    if n < 2:
        return None
    key = f"{airport}_adsb_2025-{month:02d}-{day:02d}_h{hour:02d}"
    return {
        "key": key, "airport": airport, "month": month, "day": f"2025-{month:02d}-{day:02d}", "hour": hour,
        "n": int(n), "G": int(2 * n), "lag": LAG, "delta": 2, "costs": [2, 3, 0],
        "service": donors, "probabilities": ["1/16"] * 16, "donor_days": [f"2025-{month:02d}-{d:02d}" for d in DONOR_DAYS],
        "caps": [int(n)] * HORIZON, "actual_service": evaluation,
        "data_source": "data/independent_adsb_q1/processed/airport_departures_2025_q1.csv",
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
    return {"case_id": job["case_id"], "airport": job["airport"], "month": job["month"], "day": job["day"], "hour": job["hour"], "n": case["n"], "capacity": job["capacity"], "delay": delay, "budget": budget, "saving": greedy_answer["waiting"] - exact_answer["waiting"], "exact_optimal": bool(exact_answer["optimal"]), "replay_checked": True}


def build(phase: str, airport: str, month: int) -> dict:
    prepare_processed()
    frame = load_panel(airport)
    data_root = ROOT / "data" / "independent_adsb_q1" / "cases" / airport / f"month_{month:02d}"
    results_root = ROOT / "results" / "independent_adsb_q1" / airport / f"month_{month:02d}"
    cases = []
    for day in MONTH_EVAL_DAYS[month]:
        for hour in START_HOURS:
            case = make_case(frame, airport, month, day, hour)
            if case is None:
                continue
            case_id = hashlib.sha256(case["key"].encode()).hexdigest()[:12]
            case["case_id"] = case_id
            path = data_root / f"{case_id}.json"
            save(path, case)
            cases.append({"case_id": case_id, "input_path": str(path.relative_to(ROOT)).replace("\\", "/"), "airport": airport, "month": month, "day": case["day"], "hour": hour, "capacity": "N", "divisor": 1})
    smoke_case = dict(cases[0], capacity="N4", divisor=4)
    smoke_result = evaluate_one((smoke_case, 1, 1))
    smoke = {"passed": bool(smoke_result["exact_optimal"] and smoke_result["replay_checked"]), "row": smoke_result}
    save(results_root / "smoke.json", smoke)
    if phase == "smoke":
        print(json.dumps(smoke, indent=2)); return smoke
    if not smoke["passed"]: raise RuntimeError("Q1 independent ADS-B smoke test failed")
    jobs = [dict(case, capacity=label, divisor=divisor) for case in cases for label, divisor in (("N", 1), ("N4", 4), ("N8", 8))]
    tasks = [(case, delay, budget) for case in jobs for delay in (1, 2) for budget in (1, 2)]
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(evaluate_one, tasks))
    save(results_root / "rows.json", rows)
    summary = [{"airport": airport, "month": month, "delay": delay, "budget": budget, "conditions": len([r for r in rows if r["delay"] == delay and r["budget"] == budget]), "saving": mean(r["saving"] for r in rows if r["delay"] == delay and r["budget"] == budget), "improved": sum(r["saving"] > 1e-9 for r in rows if r["delay"] == delay and r["budget"] == budget)} for delay in (1, 2) for budget in (1, 2)]
    save(results_root / "summary.json", summary)
    save(results_root / "audit.json", {"passed": all(r["exact_optimal"] and r["replay_checked"] for r in rows), "conditions": len(rows), "source_sha256": digest(PROCESSED_SOURCE)})
    print(json.dumps(summary, indent=2)); return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("phase", choices=["smoke", "full"]); parser.add_argument("--airport", choices=sorted(AIRPORTS), required=True); parser.add_argument("--month", type=int, choices=(1, 2, 3), required=True); args = parser.parse_args(); build(args.phase, args.airport, args.month)
