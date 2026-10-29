"""Summarize independent ADS-B replication results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AIRPORTS = ("ATL", "CLT", "DFW")
RESULT_ROOT = ROOT / "results" / "independent_adsb_replication"
OUT = RESULT_ROOT / "summary_all_airports.json"


def load_rows() -> pd.DataFrame:
    parts = []
    for airport in AIRPORTS:
        parts.append(pd.DataFrame(json.loads((RESULT_ROOT / airport / "rows.json").read_text(encoding="utf-8"))))
    frame = pd.concat(parts, ignore_index=True)
    frame["cluster"] = frame["airport"] + "_" + frame["day"]
    return frame


def interval(frame: pd.DataFrame, delay: int, budget: int, reps: int = 10000) -> tuple[float, float]:
    sub = frame[(frame.delay == delay) & (frame.budget == budget)]
    grouped = sub.groupby("cluster")["saving"].mean().to_numpy()
    rng = np.random.default_rng(20261029 + delay * 10 + budget)
    draws = rng.integers(0, len(grouped), size=(reps, len(grouped)))
    estimates = grouped[draws].mean(axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def main() -> None:
    frame = load_rows()
    rows = []
    for airport in AIRPORTS:
        part = frame[frame.airport == airport]
        for delay in (1, 2):
            for budget in (1, 2):
                sub = part[(part.delay == delay) & (part.budget == budget)]
                rows.append({
                    "airport": airport,
                    "delay": delay,
                    "budget": budget,
                    "conditions": int(len(sub)),
                    "saving": float(sub.saving.mean()),
                    "improved": int((sub.saving > 1e-9).sum()),
                })
    pooled = []
    for delay in (1, 2):
        for budget in (1, 2):
            sub = frame[(frame.delay == delay) & (frame.budget == budget)]
            lo, hi = interval(frame, delay, budget)
            pooled.append({
                "delay": delay,
                "budget": budget,
                "conditions": int(len(sub)),
                "saving": float(sub.saving.mean()),
                "saving_ci95_cluster": [lo, hi],
                "improved": int((sub.saving > 1e-9).sum()),
            })
    result = {
        "source": "MrAirspace aircraft-flight-schedules 2025 Q1",
        "airports": rows,
        "pooled": pooled,
        "rows": int(len(frame)),
        "audit": {
            "replay_checked": bool(frame.replay_checked.all()),
            "exact_optimal": bool(frame.exact_optimal.all()),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
