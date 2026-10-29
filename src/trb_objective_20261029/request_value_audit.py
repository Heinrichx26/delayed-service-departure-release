"""Select policies for every request valuation from saved exact cost sets.

All waiting totals use aircraft-minutes for the entire departure group.
The analysis is independent of the optimization and plotting programs.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/revision_20261029/request_value'
FRONTIER = ROOT / 'results/trb_precision_20261028/frontier_rows.json'
AIRPORTS = ('ATL', 'CLT', 'DFW')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def intervals(row):
    points = row['frontier']
    assert row['optimal'] and points[-1]['weighted_queries'] == 0
    answer = []
    for i, p in enumerate(points):
        a, c = p['weighted_waiting'], p['weighted_queries']
        lo, hi = Fraction(0), None
        for other in points:
            da = 2 * (other['weighted_waiting'] - a)
            dc = c - other['weighted_queries']
            if dc > 0:
                bound = Fraction(da, dc)
                hi = bound if hi is None else min(hi, bound)
            elif dc < 0:
                lo = max(lo, Fraction(da, dc))
        if hi is None or lo <= hi:
            answer.append(dict(index=i, lower=float(lo), upper=None if hi is None else float(hi),
                               lower_exact=str(lo), upper_exact=None if hi is None else str(hi)))
            # Independent exhaustive scalar objective evaluation inside every interval.
            test = lo + 1 if hi is None else (lo + hi) / 2
            values = [2*x['weighted_waiting'] + test*x['weighted_queries'] for x in points]
            assert values[i] == min(values)
    assert answer[0]['index'] == 0 and answer[-1]['index'] == len(points)-1
    if len(points) > 1:
        first = min(Fraction(2*(p['weighted_waiting']-points[0]['weighted_waiting']),
                             points[0]['weighted_queries']-p['weighted_queries']) for p in points[1:])
        assert Fraction(answer[0]['upper_exact']) == first
    # Verify normalizations against both saved per-departure and group measures.
    denominators = []
    for p in points:
        q = p['expected_queries']
        if q:
            denominators.append(p['weighted_queries']/q)
    denominator = denominators[0] if denominators else 16
    assert all(abs(q-denominator)<1e-9 for q in denominators)
    for p in points:
        assert abs(row['n']*p['waiting']-2*p['weighted_waiting']/denominator)<1e-8
    return dict(case_id=row['case_id'],capacity=row['capacity'],airport=row['airport'],
                day=row['day'],delay=row['delay'],width=row['width'],n=row['n'],
                points=len(points),supported=len(answer),intervals=answer,
                first_switch=answer[0]['upper'],no_request_threshold=answer[-1]['lower'])


def external(rows):
    result=[]
    for d in (1,2):
        for b in (1,2):
            group=[r for r in rows if r['delay']==d and r['budget']==b]
            assert group and all(r['exact_optimal'] and r['replay_checked'] for r in group)
            s=np.array([r['n']*r['saving'] for r in group])
            q=np.array([r['exact_requests']-r['greedy_requests'] for r in group])
            assert np.min(s)>-1e-8
            labels=sorted({(r['airport'],r['day']) for r in group})
            totals=np.array([[sum(r['n']*r['saving'] for r in group if (r['airport'],r['day'])==key),
                              sum(r['exact_requests']-r['greedy_requests'] for r in group if (r['airport'],r['day'])==key)]
                             for key in labels])
            rng=np.random.default_rng(20261029+d*10+b)
            draws=rng.integers(0,len(labels),size=(5000,len(labels)))
            sums=totals[draws].sum(axis=1)
            ratios=sums[:,0]/sums[:,1]
            positive=sums[:,1]>0
            result.append(dict(delay=d,budget=b,conditions=len(group),clusters=len(labels),
                mean_saving_per_departure=statistics.mean(r['saving'] for r in group),
                mean_group_aircraft_minutes=float(s.mean()),mean_extra_requests=float(q.mean()),
                break_even_aircraft_minutes_per_request=float(s.sum()/q.sum()),
                ci95=np.quantile(ratios[positive],[.025,.975]).tolist(),
                bootstrap_positive_denominator=int(positive.sum()),
                weak_dominance_conditions=int(np.sum(q<=1e-9))))
    return result


def main(phase):
    OUT.mkdir(parents=True,exist_ok=True)
    source=read(FRONTIER)
    paths=[ROOT/f'results/independent_adsb_replication/{a}/rows.json' for a in AIRPORTS]
    ext=[dict(r) for path in paths for r in read(path)]
    if phase=='smoke':
        selected=[next(r for r in source if r['airport']==a and r['delay']==d and r['width']==w)
                  for a in ('DEN','PHX') for d in (1,2) for w in (1,2,4)]
        # Synthetic sets exercise unsupported choices and exact ties.
        for pairs in [[(0,4),(2,3),(3,2),(8,0)],[(0,2),(2,1),(4,0)],[(3,0)]]:
            p=[dict(weighted_waiting=a,weighted_queries=c,waiting=2*a/16,expected_queries=c/16) for a,c in pairs]
            intervals(dict(frontier=p,optimal=True,n=1,case_id='synthetic',capacity='N',airport='X',day='0',delay=1,width=1))
        ext=[r for r in ext if r['day']==min(x['day'] for x in ext if x['airport']==r['airport'])]
    else:
        smoke=read(OUT/'smoke.json')
        assert smoke['passed'] and smoke['script_sha256']==sha(Path(__file__))
        selected=source
    rows=[intervals(r) for r in selected]
    summary=[]
    for d in (1,2):
        for w in (1,2,4):
            g=[r for r in rows if r['delay']==d and r['width']==w]
            active=[r for r in g if r['points']>1]
            first=[r['first_switch'] for r in active]
            zero=[r['no_request_threshold'] for r in active]
            summary.append(dict(delay=d,width=w,conditions=len(g),active=len(active),
                points=sum(r['points'] for r in g),supported=sum(r['supported'] for r in g),
                first_switch_median=statistics.median(first),first_switch_range=[min(first),max(first)],
                no_request_median=statistics.median(zero),no_request_range=[min(zero),max(zero)]))
    result=dict(passed=True,phase=phase,script_sha256=sha(Path(__file__)),
        inputs={str(p.relative_to(ROOT)):sha(p) for p in [FRONTIER]+paths},
        conditions=len(rows),points=sum(r['points'] for r in rows),
        supported=sum(r['supported'] for r in rows),summary=summary,external=external(ext))
    save(OUT/f'{phase}.json',result)
    save(OUT/f'{phase}_policy_intervals.json',rows)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('smoke','full'))
    main(parser.parse_args().phase)
