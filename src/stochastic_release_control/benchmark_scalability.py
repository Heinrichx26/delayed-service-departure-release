"""Benchmark exact recursion states and cold solve time on public cases."""
from __future__ import annotations

import glob
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from math import ceil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
from .joint_replication.extensions import old
from .joint_replication.joint_design import JointDesign
from .joint_replication.validate_joint import replay


def evaluate(task: tuple[str, int, int, int]) -> dict:
    path, divisor, delay, budget = task
    case = json.loads(Path(path).read_text(encoding="utf-8"))
    cap = ceil(case["n"] / divisor)
    case["caps"] = [cap] * len(case["caps"])
    problem = old.prepare(case)
    start = time.perf_counter()
    solver = JointDesign(problem, case["lag"] + 2, delay, budget)
    answer = solver.solve()
    elapsed_ms = 1000.0 * (time.perf_counter() - start)
    replay(problem, solver.replay(), case["lag"] + 2, delay, budget)
    return {
        "airport": case["airport"],
        "case_id": case["case_id"],
        "n": case["n"],
        "divisor": divisor,
        "delay": delay,
        "budget": budget,
        "states": answer["states"],
        "runtime_ms": elapsed_ms,
        "exact_optimal": bool(answer["optimal"]),
        "replay_checked": True,
    }


def main() -> None:
    files = []
    for airport in ("ATL", "CLT", "DFW"):
        files.extend(glob.glob(str(ROOT / "data" / "independent_adsb" / "cases" / airport / "*.json")))
    tasks = [(path, divisor, delay, budget)
             for path in files
             for divisor in (1, 4, 8)
             for delay in (1, 2)
             for budget in (1, 2)]
    rows = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for row in pool.map(evaluate, tasks):
            rows.append(row)
    out = ROOT / "results" / "independent_adsb_replication" / "scalability_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({
        "rows": len(rows),
        "exact_optimal": all(row["exact_optimal"] for row in rows),
        "replay_checked": all(row["replay_checked"] for row in rows),
        "max_runtime_ms": max(row["runtime_ms"] for row in rows),
        "max_states": max(row["states"] for row in rows),
    }, indent=2))


if __name__ == "__main__":
    main()
