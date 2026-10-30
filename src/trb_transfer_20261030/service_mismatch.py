"""Takeoff-loss bounds for a release preserving a reference service calendar."""
from itertools import product
from pathlib import Path
import hashlib
import json
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src/trb_error_20261030'))
from robust_reports import physical


def oscillation(actual, donor):
    """Range of cumulative count differences, including the empty prefix."""
    total = low = high = 0
    answer = []
    for a, s in zip(actual, donor):
        total += a-s
        low, high = min(low, total), max(high, total)
        answer.append(high-low)
    return tuple(answer)


def bound(releases, actual, problem):
    earliest = []
    previous = 0
    for cap in problem['caps']:
        previous = min(problem['n'], previous+cap)
        earliest.append(previous)
    ref = physical(earliest, actual, problem['lag'])
    done = physical(releases, actual, problem['lag'])
    loss = tuple(a-b for a, b in zip(ref, done))
    preserving = [i for i, b in enumerate(problem['bounds'])
                  if all(u >= x for u, x in zip(releases, b))]
    assert preserving, 'Release must preserve at least one donor calendar'
    bounds = [oscillation(actual, problem['paths'][i]) for i in preserving]
    upper = tuple(min(ref[t], min(b[t] for b in bounds)) for t in range(len(ref)))
    assert all(0 <= a <= b for a, b in zip(loss, upper))
    return dict(loss=list(loss), upper=list(upper), preserving_donors=len(preserving),
                reference=list(ref), departures=list(done))


def smoke():
    """Enumerate releases and calendars independently of the request optimizer."""
    checked = positive = tight_positive = 0
    for h, n, lag in [(5, 2, 1), (6, 3, 2)]:
        caps = (1,)*h
        earliest = tuple(min(n, t+1) for t in range(h))
        schedules = []
        for increments in product((0, 1), repeat=h):
            if sum(increments) != n:
                continue
            schedules.append(tuple(sum(increments[:t+1]) for t in range(h)))
        calendars = list(product((0, 1, 2), repeat=h))
        rng = random.Random(20261030+h)
        # All actual calendars in the shorter case; fixed broad sample otherwise.
        for donor in calendars:
            ref = physical(earliest, donor, lag)
            if ref[-1] != n:
                continue
            for release in schedules:
                if physical(release, donor, lag) != ref:
                    continue
                actuals = calendars if h == 5 else rng.sample(calendars, 12)
                for actual in actuals:
                    a = physical(earliest, actual, lag)
                    b = physical(release, actual, lag)
                    upper = oscillation(actual, donor)
                    for t, (x, y, z) in enumerate(zip(a, b, upper)):
                        assert 0 <= x-y <= min(x, z)
                        positive += x > y
                        tight_positive += x > y and x-y == z
                    checked += 1
    result = dict(passed=True, trajectory_comparisons=checked,
                  positive_slot_losses=positive, positive_equalities=tight_positive,
                  seed=20261030, formula='Range of cumulative actual-minus-donor service counts')
    out = ROOT/'results/trb_transfer_20261030/bound_smoke.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    smoke()
