"""Small independent checks of interval releases and calendar error bounds.

This file does not import or modify the frozen acquisition implementation.
The smoke phase must pass before the exhaustive phase is permitted.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/revision_20261028/substantive_strengthening'
SOURCE = Path(__file__)


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def digest():
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def earliest(n, caps):
    u, result = 0, []
    for k in caps:
        u = min(n, u+k)
        result.append(u)
    return tuple(result)


def physical(u, c, lag):
    done, inventory = [], []
    previous = 0
    for t, service in enumerate(c):
        arrived = u[t-lag] if t >= lag else 0
        previous = min(arrived, previous+service)
        done.append(previous)
        inventory.append(u[t]-previous)
    return tuple(done), tuple(inventory)


def deadlines(n, caps, lag, c):
    h = len(caps)
    ref, _ = physical(earliest(n, caps), c, lag)
    required = [ref[min(v+lag, h-1)] for v in range(h)]
    closed = required[:]
    for t in range(h-2, -1, -1):
        closed[t] = max(closed[t], closed[t+1]-caps[t+1])
    return ref, tuple(closed)


def target(n, caps, lag, c, prefix=()):
    ref, bound = deadlines(n, caps, lag, c)
    result = list(prefix)
    before = prefix[-1] if prefix else 0
    for t in range(len(prefix), len(caps)):
        before = max(before, bound[t])
        result.append(before)
    return tuple(result)


def feasible_release(u, n, caps):
    previous = 0
    for x, cap in zip(u, caps):
        if not 0 <= x-previous <= cap:
            return False
        previous = x
    return previous == n


def error_budget(actual, nominal):
    error = 0
    lowest = 0
    highest_rise = 0
    result = []
    for c, q in zip(actual, nominal):
        error += c-q
        highest_rise = max(highest_rise, error-lowest)
        lowest = min(lowest, error)
        result.append(highest_rise)
    return tuple(result)


def check_error(n, caps, lag, nominal, actual, releases):
    nominal_ref, _ = deadlines(n, caps, lag, nominal)
    assert physical(releases, nominal, lag)[0] == nominal_ref
    actual_ref, _ = deadlines(n, caps, lag, actual)
    actual_done, _ = physical(releases, actual, lag)
    budget = error_budget(actual, nominal)
    losses = [a-d for a, d in zip(actual_ref, actual_done)]
    assert all(0 <= loss <= b for loss, b in zip(losses, budget))
    return max(losses), max(budget), any(loss == b and b > 0 for loss,b in zip(losses,budget))


def clipped_box_paths(lo, hi, n, caps, lag):
    return tuple(c for c in product(*(range(a,b+1) for a,b in zip(lo,hi)))
                 if deadlines(n,caps,lag,c)[0][-1] == n)


def check_box(n, caps, lag, lo, hi, lead):
    h = len(caps)
    paths = clipped_box_paths(lo, hi, n, caps, lag)
    upper_complete = deadlines(n, caps, lag, hi)[0][-1] == n
    assert bool(paths) == upper_complete
    if not paths:
        return dict(paths=0,corner_checks=1,policy_slots=0,regret_checks=0)
    full_targets = {c: target(n,caps,lag,c) for c in paths}
    corners, policies, regrets = 0, 0, 0
    for actual in paths:
        u, historical_width, widths = [], 0, []
        for t in range(h):
            observed_end = min(h,max(0,t+lead+1))
            lower = tuple(actual[j] if j < observed_end else lo[j] for j in range(h))
            upper = tuple(actual[j] if j < observed_end else hi[j] for j in range(h))
            compatible = [c for c in paths if c[:observed_end] == actual[:observed_end]]
            assert compatible and upper in compatible
            envelope = deadlines(n,caps,lag,upper)[1][t]
            assert envelope == max(deadlines(n,caps,lag,c)[1][t] for c in compatible)
            exact_max = max(full_targets[c][t] for c in compatible)
            relaxed_min = target(n,caps,lag,lower)[t]
            assert exact_max == target(n,caps,lag,upper)[t]
            assert relaxed_min <= min(full_targets[c][t] for c in compatible)
            width = exact_max-relaxed_min
            widths.append(width)
            historical_width = max(historical_width,width)
            value = max(u[-1] if u else 0,envelope)
            u.append(value)
            assert 0 <= value-full_targets[actual][t] <= historical_width
            corners += 1
        assert feasible_release(u,n,caps)
        ref = deadlines(n,caps,lag,actual)[0]
        done, inventory = physical(u,actual,lag)
        assert done == ref
        regret = sum(value-star for value,star in zip(u,full_targets[actual]))
        running, bound = 0, 0
        for width in widths:
            running = max(running,width)
            bound += running
        assert 0 <= regret <= bound
        policies += h
        regrets += 1
    return dict(paths=len(paths),corner_checks=corners,policy_slots=policies,regret_checks=regrets)


def smoke():
    # An extra early opportunity is missed by the nominal late release.
    caps = (1,1,1,1)
    n, lag = 1, 0
    nominal, actual = (0,0,0,1),(0,1,0,0)
    u = target(n,caps,lag,nominal)
    sharp = check_error(n,caps,lag,nominal,actual,u)
    assert sharp[0] == sharp[1] == 1 and sharp[2]
    # Lower service cannot create a completion loss for a nominal-preserving U.
    removed = check_error(1,caps,0,(0,1,0,1),(0,0,0,1),target(1,caps,0,(0,1,0,1)))
    assert removed[:2] == (0,0)
    # The lower corner may be incomplete; intersecting with completion suffices.
    box = check_box(2,(1,1,1,1),1,(0,0,0,0),(0,2,2,2),-1)
    assert box['paths'] > 1
    save('interval_theory_smoke.json',dict(passed=True,source_sha256=digest(),
                                         sharp_error=sharp,capacity_decrease=removed,box=box))
    print(dict(passed=True,sharp_error=sharp,capacity_decrease=removed,box=box),flush=True)


def exhaustive():
    smoke_report = json.loads((OUT/'interval_theory_smoke.json').read_text(encoding='utf-8'))
    assert smoke_report['passed'] and smoke_report['source_sha256'] == digest()
    begin = perf_counter()
    error_checks, sharp_checks, box_cases = 0,0,0
    totals = dict(paths=0,corner_checks=0,policy_slots=0,regret_checks=0)
    for h in (3,4):
        for caps in product((0,1),repeat=h):
            for n in (1,2):
                if sum(caps) < n:
                    continue
                releases = [u for u in product(range(n+1),repeat=h) if feasible_release(u,n,caps)]
                for lag in (0,1):
                    calendars = [c for c in product((0,1),repeat=h)
                                 if deadlines(n,caps,lag,c)[0][-1] == n]
                    for nominal in calendars:
                        ref = deadlines(n,caps,lag,nominal)[0]
                        preserving = [u for u in releases if physical(u,nominal,lag)[0] == ref]
                        for actual in product((0,1),repeat=h):
                            for u in preserving:
                                _,_,sharp = check_error(n,caps,lag,nominal,actual,u)
                                error_checks += 1
                                sharp_checks += sharp
                    for hi in product((0,1),repeat=h):
                        lo = (0,)*h
                        for lead in (-1,0,1,h-1):
                            checked = check_box(n,caps,lag,lo,hi,lead)
                            box_cases += 1
                            for key,value in checked.items():
                                totals[key] += value
    report = dict(passed=True,source_sha256=digest(),error_checks=error_checks,
                  sharp_error_cases=sharp_checks,box_cases=box_cases,**totals,
                  elapsed_seconds=perf_counter()-begin)
    save('interval_theory_exhaustive.json',report)
    print(report,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['smoke','exhaustive'],required=True)
    args = parser.parse_args()
    globals()[args.phase]()
