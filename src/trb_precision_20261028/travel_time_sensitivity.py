"""Exact travel-time sensitivity for the departure information problem.

The service calendars, probabilities and release capacities remain fixed while
the gate-to-runway travel lag changes. Each run is replayed independently.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from math import ceil
from statistics import mean

from band_design import BandDesign, ROOT, old
from fixed_band import fixed_band
from frontier_design import FrontierDesign
from validate_band import replay

DATA = ROOT / 'data/trb_precision_20261028'
OUT = ROOT / 'results/trb_precision_20261028/travel_time_sensitivity'
LAGS = (4, 5, 6)


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2), encoding='utf-8')


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def one(job):
    case = read(ROOT / job['input_path'])
    case = dict(case, lag=job['lag'],
                caps=[ceil(case['n'] / job['divisor'])] * len(case['caps']))
    p = old.prepare(case)
    if p is None:
        return dict(case_id=job['case_id'], lag=job['lag'], complete=False)
    lead = case['lag'] + 2
    adaptive = FrontierDesign(p, lead, 1, 1, 2, seconds_limit=60)
    frontier = adaptive.solve()
    assert frontier['optimal'] if 'optimal' in frontier else True
    pair = (frontier['frontier'][0]['weighted_waiting'],
            frontier['frontier'][0]['weighted_queries'])
    replay(p, adaptive.replay_pair(pair), lead, 1, 1, 2)
    fixed = fixed_band(p, lead, 1, 1, 2)
    assert fixed['optimal']
    replay(p, fixed['paths'], lead, 1, 1, 2)
    return dict(case_id=job['case_id'], airport=job['airport'], day=job['day'],
                capacity=job['capacity'], lag=job['lag'], n=case['n'], complete=True,
                adaptive_waiting=frontier['frontier'][0]['waiting'],
                adaptive_requests=frontier['frontier'][0]['expected_queries'],
                fixed_waiting=fixed['waiting'],
                fixed_requests=fixed['count'],
                adaptive_gain=fixed['waiting']-frontier['frontier'][0]['waiting'],
                states=frontier['states'])


def jobs():
    all_jobs = read(DATA / 'panel_design.json')['jobs']
    selected = [j for j in all_jobs if j['capacity'] == 'N']
    # One window per airport-date pair keeps the sensitivity distinct from
    # the main pooled comparison while retaining both reconstructed airports.
    chosen = []
    seen = set()
    for j in selected:
        key = (j['airport'], j['day'])
        if key not in seen:
            case = read(ROOT / j['input_path'])
            if old.prepare(dict(case, lag=6,
                                caps=[ceil(case['n'] / j['divisor'])] * len(case['caps']))) is not None:
                chosen.append(j)
            seen.add(key)
    return [dict(j, lag=lag) for j in chosen for lag in LAGS]


def run(phase):
    js = jobs()
    if phase == 'smoke':
        js = js[:3]
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(one, js))
    by_lag=[]
    for lag in LAGS:
        g=[r for r in rows if r['lag']==lag and r['complete']]
        by_lag.append(dict(lag=lag, conditions=len(g), attempted=sum(r['lag']==lag for r in rows),
                           adaptive_waiting=mean(r['adaptive_waiting'] for r in g) if g else None,
                           fixed_waiting=mean(r['fixed_waiting'] for r in g) if g else None,
                           adaptive_gain=mean(r['adaptive_gain'] for r in g) if g else None,
                           adaptive_requests=mean(r['adaptive_requests'] for r in g) if g else None,
                           max_states=max(r['states'] for r in g) if g else None))
    result=dict(passed=True, phase=phase, lags=LAGS, rows=rows,
                summary=by_lag,
                source_sha256=digest(DATA/'panel_design.json'))
    save(OUT/(phase+'.json'), result)
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('phase', choices=('smoke','full'))
    run(parser.parse_args().phase)
