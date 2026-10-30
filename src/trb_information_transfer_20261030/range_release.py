"""Minimum cumulative release over integer count ranges and shared-error reports."""
from functools import lru_cache
from itertools import product
from pathlib import Path
import json
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src/trb_error_20261030'))
from robust_reports import physical


@lru_cache(maxsize=250000)
def requirement(upper, n, caps, lag, past, reports, gamma=1):
    """Return exact robust requirements; None means inconsistent observations.

    reports is a sorted tuple of (service slot, returned integer count).
    For each cumulative error, only the largest reference departure count
    is needed, because the physical service update is monotone in that count.
    """
    h = len(upper)
    t = len(past)
    given = dict(reports)
    if any(c < 0 or c > upper[j] for j, c in enumerate(past)):
        return None
    early_list=[]
    cumulative=0
    for cap in caps:
        cumulative=min(n,cumulative+cap)
        early_list.append(cumulative)
    earliest=tuple(early_list)
    minimum_error = []
    for j in range(h):
        if j not in given:
            minimum_error.append(0)
        elif j < t:
            minimum_error.append(abs(past[j]-given[j]))
        else:
            minimum_error.append(max(0, given[j]-upper[j], -given[j]))
    if sum(minimum_error) > gamma:
        return None
    suffix = [0]*(h+1)
    for j in range(h-1, -1, -1):
        suffix[j] = suffix[j+1]+minimum_error[j]
    states = {0: 0}
    max_reference = []
    for j in range(h):
        arrived = earliest[j-lag] if j >= lag else 0
        if j < t:
            values = [(past[j], abs(past[j]-given[j]) if j in given else 0)]
        elif j not in given:
            values = [(upper[j], 0)]
        else:
            y = given[j]
            values = [(v, abs(v-y)) for v in range(max(0,y-gamma), min(upper[j],y+gamma)+1)]
        following = {}
        for used, done in states.items():
            for count, error in values:
                cost = used+error
                if cost+suffix[j+1] <= gamma:
                    value = min(arrived, done+count)
                    following[cost] = max(following.get(cost, -1), value)
        if not following:
            return None
        states = following
        max_reference.append(max(states.values()))
    needed = [max_reference[min(j+lag,h-1)] for j in range(h)]
    needed[-1] = n
    for j in range(h-2,-1,-1):
        needed[j] = max(needed[j], needed[j+1]-caps[j+1])
    return tuple(needed)


def exhaustive_check():
    rng = random.Random(20261030)
    cases = 0
    trajectories = 0
    for _ in range(160):
        h = rng.choice((4,5,6))
        n = rng.choice((2,3,4))
        lag = rng.choice((0,1,2))
        upper = tuple(rng.randint(0,2) for _ in range(h))
        cap = rng.choice((1,2,n))
        caps = (cap,)*h
        early = tuple(min(n,(j+1)*cap) for j in range(h))
        actual = tuple(rng.randint(0,u) for u in upper)
        t = rng.randrange(h)
        indices = rng.sample(range(h),rng.randrange(min(3,h)+1))
        reports = tuple(sorted((j,max(0,min(n,actual[j]+rng.choice((-1,0,1))))) for j in indices))
        gamma = rng.choice((0,1,2))
        predicted = requirement(upper,n,caps,lag,actual[:t],reports,gamma)
        compatible = []
        for c in product(*(range(u+1) for u in upper)):
            if c[:t] != actual[:t] or sum(abs(c[j]-y) for j,y in reports)>gamma:
                continue
            ref = physical(early,c,lag)
            b = [ref[min(j+lag,h-1)] for j in range(h)]
            b[-1] = n
            for j in range(h-2,-1,-1):
                b[j] = max(b[j],b[j+1]-caps[j+1])
            compatible.append(tuple(b))
            trajectories += 1
        expected = tuple(max(b[j] for b in compatible) for j in range(h)) if compatible else None
        assert predicted == expected, (upper,predicted,expected)
        cases += 1
    result = dict(passed=True,cases=cases,compatible_trajectories=trajectories,
                  checks='Exact equality with exhaustive integer-calendar release bounds, including incomplete groups and inconsistent reports')
    out = ROOT/'results/trb_information_transfer_20261030'
    out.mkdir(parents=True,exist_ok=True)
    (out/'range_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    exhaustive_check()
