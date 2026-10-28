"""Independent finite-domain audit of fixed-bin report inference.

This audit enumerates the values in a union of integer intervals and
checks the frozen inference helper against that direct response set.
It does not rerun the held-out experiment or alter its frozen sources.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations_with_replacement
import json
from pathlib import Path

import interval_precision_v2 as frozen

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/revision_20261028/substantive_strengthening'
SOURCE = Path(__file__).resolve()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected(boxes, width, n, available, known):
    answer = {}
    for j in available:
        if j in known:
            continue
        responses = set()
        for lo, hi in boxes:
            for value in range(lo[j], hi[j] + 1):
                lower = (value // width) * width
                responses.add((lower, min(n, lower + width - 1)))
        if len(responses) == 1:
            answer[j] = responses.pop()
    return answer


def check(boxes, width, n, available, known):
    got = frozen.inferable_reports(boxes, width, n, available, known)
    assert got == expected(boxes, width, n, available, known)
    for j, (lower, upper) in got.items():
        assert all(lower <= lo[j] <= hi[j] <= upper for lo, hi in boxes)
    # Inference leaves every integer box unchanged, hence leaves every
    # upper-corner release bound unchanged under every release model.
    return len(got)


def run(phase):
    manifest = dict(audit_source_sha256=digest(SOURCE),
                    frozen_source_sha256=digest(Path(frozen.__file__)))
    if phase == 'smoke':
        cases = [([([0], [1])], 2, 3, [0], {}),
                 ([([1], [2])], 2, 3, [0], {}),
                 ([([0], [1]), ([2], [2])], 2, 3, [0], {}),
                 ([([0], [1])], 2, 3, [0], {0: (0, 1)}),
                 ([([0, 1], [1, 3])], 2, 3, [0, 1], {})]
        counts = [check(*case) for case in cases]
        assert counts == [1, 0, 0, 0, 1]
        report = dict(passed=True, phase=phase, **manifest, cases=len(cases))
    else:
        smoke = json.loads((OUT / 'interval_inference_smoke.json').read_text(encoding='utf-8'))
        assert smoke['passed'] and all(smoke[k] == v for k, v in manifest.items())
        cases = inferred = 0
        for n in range(1, 9):
            intervals = [(lo, hi) for lo in range(n + 1) for hi in range(lo, n + 1)]
            for first, second in combinations_with_replacement(intervals, 2):
                boxes = [([lo], [hi]) for lo, hi in (first, second)]
                for width in range(1, 6):
                    for available, known in (([0], {}), ([], {}), ([0], {0: (0, 0)})):
                        inferred += check(boxes, width, n, available, known)
                        cases += 1
        report = dict(passed=True, phase=phase, **manifest,
                      cases=cases, inferred_reports_checked=inferred,
                      domain='N=1..8, widths=1..5, every two-interval union; available, unavailable and known filtering',
                      interpretation='Endpoint agreement exactly identifies a constant bin over the untrimmed box union; every inferred interval leaves that union unchanged.')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'interval_inference_{phase}.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['smoke', 'exhaustive'], required=True)
    run(parser.parse_args().phase)
