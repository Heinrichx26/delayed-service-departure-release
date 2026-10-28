"""Joint departure decisions with delayed, coarsened service reports.

Realized past counts remain exact. Only requested future counts are binned.
The reporting width counts possible integer service values, not aircraft error.
"""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/trb_joint_20261028'))
from joint_design import JointDesign, old, StateLimit


class BandDesign(JointDesign):
    def __init__(self, problem, lead, delay, budget, width=1, **limits):
        assert isinstance(width, int) and width >= 1
        self.width = width
        super().__init__(problem, lead, delay, budget, **limits)
        self.report_part = lru_cache(None)(self.report_part)

    def report_part(self, mask, slots):
        groups = defaultdict(int)
        for s, path in enumerate(self.paths):
            if mask >> s & 1:
                key = tuple(path[j] // self.width for j in slots)
                groups[key] |= 1 << s
        return tuple(sorted(groups.values()))

    def canonical(self, mask, pending):
        return tuple(sorted((due, j) for due, j in pending
                            if len(self.report_part(mask, (j,))) > 1))

    def candidates(self, t, mask, pending):
        existing = {j for _, j in pending}
        seen = set()
        result = []
        for j in range(t + self.delay, min(self.h, t + self.lead + 1)):
            if j in existing:
                continue
            partition = self.report_part(mask, (j,))
            if len(partition) > 1 and partition not in seen:
                seen.add(partition)
                result.append(j)
        return tuple(result)

    def transition(self, t, mask, pending, request):
        combined = tuple(pending) + tuple((t + self.delay, j) for j in request)
        due = tuple(j for arrival, j in combined if arrival == t + 1)
        future = tuple((arrival, j) for arrival, j in combined if arrival > t + 1)
        groups = defaultdict(int)
        for s, path in enumerate(self.paths):
            if mask >> s & 1:
                # Service at t becomes observed; confirmations can remain imprecise.
                key = (path[t],) + tuple(path[j] // self.width for j in due)
                groups[key] |= 1 << s
        return tuple((sub, self.canonical(sub, future))
                     for sub in sorted(groups.values()))

