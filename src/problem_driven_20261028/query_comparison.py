"""Greedy and optimal confirmations with identical physical/action targets."""
from __future__ import annotations

import argparse
from fractions import Fraction
import gzip
import json
from pathlib import Path
from time import perf_counter

import info_design as design
from experiment import OLD, csv_write

ROOT, DATA, OUT = design.ROOT, design.DATA, design.OUT


def greedy(problem, lead):
    paths, targets = problem['paths'], problem['target']
    weights, t0, h = problem['weights'], problem['t0'], problem['horizon']

    def groups(members, j):
        bins = {}
        for s in members:
            bins.setdefault(paths[s][j], []).append(s)
        return list(bins.values())

    def edges(members,t):
        return sum(weights[s]*weights[q] for i,s in enumerate(members)
                   for q in members[:i] if targets[s][t] != targets[q][t])

    def choose(members,t):
        mass=sum(weights[s] for s in members)
        initial=edges(members,t)
        candidates=[]
        for j in range(t0,h):
            if max(t0,j-lead)>t:
                continue
            parts=groups(members,j)
            if len(parts)<=1:
                continue
            # Expected removed pair weight; multiply by mass for exact arithmetic.
            score=mass*initial-sum(sum(weights[s] for s in part)*edges(part,t) for part in parts)
            candidates.append((score,-j,j))
        return max(candidates)[2] if candidates else None

    records=[]
    for s,path in enumerate(paths):
        members=[q for q in range(len(paths)) if paths[q][:t0]==path[:t0]]
        actual=list(problem['prefix'])
        queries=[]
        for t in range(t0,h):
            while len({targets[q][t] for q in members})>1:
                j=choose(members,t)
                if j is None:
                    return dict(feasible=False)
                queries.append(dict(t=t,j=j,value=path[j]))
                members=[q for q in members if paths[q][j]==path[j]]
            actual.append(targets[s][t])
            members=[q for q in members if paths[q][t]==path[t]]
        assert actual==targets[s]
        assert all(0<=u<=k for u,k in zip(design.base.increments(actual),problem['caps']))
        completed,inventory,_=design.base.physical(actual,path,problem['n'],problem['lag'])
        assert completed==problem['references'][s]
        records.append(dict(s=s,queries=queries,count=len(queries),
                            completions=completed,waiting=problem['case']['delta']*sum(inventory[t0:])))
    weighted=sum(weights[r['s']]*r['count'] for r in records)
    return dict(feasible=True,expected_queries=float(Fraction(weighted,problem['denominator'])),
                expected_queries_exact=str(Fraction(weighted,problem['denominator'])),replay=records)


def execute(smoke):
    manifest=design.read(OLD)
    selected=[row for row in manifest['rows'] if not smoke or row['key'] in manifest['smoke_keys']]
    records,detailed=[],[]
    start=perf_counter()
    exact_by_key={}
    import csv
    with (OUT/'frontiers.csv').open(encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            exact_by_key[(row['case_id'],int(row['t0']),int(row['lead']))]=row
    for row in selected:
        path=ROOT/row['input_path']
        assert design.digest(path)==row['input_sha256']
        case=design.read(path)
        empty=design.prepare(case)
        if empty is None:
            continue
        prefix=design.base.frontier(empty['paths'],empty['bounds'],[],-1)[0][:1]
        for executed in ([],prefix):
            problem=design.prepare(case,executed)
            for name,lead in (('transport_plus_two',case['lag']+2),('full_horizon',len(case['caps'])-1)):
                result=greedy(problem,lead)
                optimal=exact_by_key[(row['case_id'],len(executed),lead)]
                assert result['feasible'] and optimal['feasible']=='True'
                optimum=float(optimal['expected_queries'])
                assert result['expected_queries']>=optimum-1e-12
                records.append(dict(case_id=row['case_id'],key=row['key'],capacity_label=row['capacity_label'],
                                    t0=len(executed),lead=lead,lead_name=name,
                                    greedy_queries=result['expected_queries'],optimal_queries=optimum,
                                    static_queries=float(optimal['static_queries']),
                                    strict=result['expected_queries']>optimum))
                detailed.append(dict(case_id=row['case_id'],t0=len(executed),lead=lead,**result))
    return dict(passed=True,smoke=smoke,rows=records,replay=detailed,
                source_sha256=design.digest(Path(__file__)),elapsed_seconds=perf_counter()-start)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['smoke','panel'],required=True)
    args=parser.parse_args()
    if args.phase=='smoke':
        result=execute(True)
        design.save(OUT/'greedy_smoke.json',result)
    else:
        smoke=design.read(OUT/'greedy_smoke.json')
        assert smoke['passed'] and smoke['source_sha256']==design.digest(Path(__file__))
        design.save(DATA/'greedy_design.json',dict(date='2026-10-28',frozen_before_panel=True,
                    input_manifest_sha256=design.digest(OLD),smoke_sha256=design.digest(OUT/'greedy_smoke.json'),
                    exact_panel_sha256=design.digest(OUT/'panel.json'),
                    source_sha256=design.digest(Path(__file__)),leads=['transport_plus_two','full_horizon'],
                    comparator='Greedy expected removal of probability-weighted pairs with different current target releases; exact arithmetic; earliest-slot ties.',
                    prefixes=[0,1],coverage='Every complete support in the 282 candidates.'))
        result=execute(False)
        csv_write(OUT/'greedy_comparison.csv',result['rows'])
        (OUT/'greedy_replay.json.gz').write_bytes(gzip.compress(json.dumps(result,separators=(',',':')).encode(),mtime=0))
        design.save(OUT/'greedy_summary.json',{k:v for k,v in result.items() if k not in ('rows','replay')})
    print(dict(passed=True,smoke=result['smoke'],settings=len(result['rows']),
               strict=sum(row['strict'] for row in result['rows']),elapsed_seconds=result['elapsed_seconds']),flush=True)


if __name__=='__main__':
    main()
