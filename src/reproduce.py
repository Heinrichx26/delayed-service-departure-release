"""Run a small independent verification before a full numerical panel."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/trb_precision_20261028'))
from band_design import BandDesign, old
from validate_band import brute, replay


def smoke():
    started = perf_counter()
    design = json.loads((ROOT / 'data/substantive_20261028/holdout_boxes/panel_design.json').read_text())
    assert len(design['rows']) == 94
    for row in design['rows']:
        path = ROOT / row['input_path'].replace('\\', '/')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['input_sha256']
    case = dict(n=2, lag=1, caps=[1]*6,
                service=[[0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 1],
                         [0, 0, 0, 0, 1, 1]],
                probabilities=['1/6', '2/6', '3/6'], delta=2)
    conditions = 0
    for prefix in [(), (1,)]:
        problem = old.prepare(case, prefix=prefix)
        for delay in [1, 2]:
            for budget in [1, 2]:
                for width in [1, 2, 4]:
                    model = BandDesign(problem, 3, delay, budget, width)
                    result = model.solve()
                    oracle = brute(problem, 3, delay, budget, width)
                    assert (result['weighted_waiting'], result['weighted_queries']) == oracle
                    replay(problem, model.replay(), 3, delay, budget, width)
                    conditions += 1
    output = dict(passed=True, numerical_conditions=conditions,
                  hashed_evaluation_cases=len(design['rows']),
                  seconds=perf_counter()-started)
    destination = ROOT / 'results/public_verification'
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'smoke.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['smoke'])
    parser.parse_args()
    smoke()
