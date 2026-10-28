"""Static confirmation-budget waiting frontier, independently solved by MILP."""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import gzip
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

import info_design as design
from experiment import OLD, csv_write

ROOT,DATA,OUT=design.ROOT,design.DATA,design.OUT


def menu_wait(problem,lead,slots):
    paths=problem['paths']
    plans=[list(problem['prefix']) for _ in paths]
    for t in range(problem['t0'],problem['horizon']):
        groups={}
        delivered=[j for j in slots if max(problem['t0'],j-lead)<=t]
        for s,path in enumerate(paths):
            groups.setdefault((tuple(path[:t]),tuple(path[j] for j in delivered)),[]).append(s)
        for members in groups.values():
            before=plans[members[0]][-1] if t else 0
            assert len({tuple(plans[s]) for s in members})==1
            u=max(before,max(problem['bounds'][s][t] for s in members))
            for s in members:
                plans[s].append(u)
    path_wait=[]
    for s,path in enumerate(paths):
        assert all(0<=u<=k for u,k in zip(design.base.increments(plans[s]),problem['caps']))
        completed,inventory,_=design.base.physical(plans[s],path,problem['n'],problem['lag'])
        assert completed==problem['references'][s]
        path_wait.append(problem['case']['delta']*sum(inventory[problem['t0']:]))
    value=sum(Fraction(w,problem['denominator'])*v for w,v in zip(problem['weights'],path_wait))
    return value,plans


def solve(problem,lead,budget):
    start=perf_counter()
    paths=problem['paths']
    n,h,t0=problem['n'],problem['horizon'],problem['t0']
    menu=design.menu_cover(problem,lead)
    assert menu['feasible']
    if budget==0 or budget>=menu['optimal_message_count']:
        selected=[] if budget==0 else menu['selected_slots']
        value,plans=menu_wait(problem,lead,selected)
        return dict(budget=budget,waiting=float(value),waiting_exact=str(value),selected_slots=selected,
                    method='exact_frontier_endpoint',gap=0,elapsed_seconds=perf_counter()-start)
    # Retain every service slot varying within at least one naturally unresolved
    # pair while available. This includes messages useful only below full recovery.
    candidates=[j for j in range(t0,h) if any(paths[s][j]!=paths[q][j]
                 and max(t0,j-lead)<=t and paths[s][:t]==paths[q][:t]
                 for s in range(len(paths)) for q in range(s) for t in range(t0,h))]
    z=len(paths)*h
    columns=z+len(candidates)
    lower=np.zeros(columns)
    upper=np.array([n]*z+[1]*len(candidates),dtype=float)
    costs=np.zeros(columns)
    for s in range(len(paths)):
        for t in range(h):
            index=s*h+t
            lower[index]=problem['references'][s][min(t+problem['lag'],h-1)]
            if t<t0:
                lower[index]=upper[index]=problem['prefix'][t]
            else:
                costs[index]=problem['case']['delta']*problem['weights'][s]/problem['denominator']
    rr,cc,vv,lo,hi=[],[],[],[],[]
    def add(pairs,l,u):
        r=len(lo)
        for col,v in pairs:
            rr.append(r);cc.append(col);vv.append(v)
        lo.append(l);hi.append(u)
    for s in range(len(paths)):
        for t in range(h):
            add([(s*h+t,1)]+([(s*h+t-1,-1)] if t else []),0,problem['caps'][t])
    for t in range(t0,h):
        for s in range(len(paths)):
            for q in range(s):
                if paths[s][:t]!=paths[q][:t]:
                    continue
                selected=[(z+i,-n) for i,j in enumerate(candidates)
                          if j-lead<=t and paths[s][j]!=paths[q][j]]
                add([(s*h+t,1),(q*h+t,-1)]+selected,-np.inf,0)
                add([(s*h+t,-1),(q*h+t,1)]+selected,-np.inf,0)
    add([(z+i,1) for i in range(len(candidates))],0,budget)
    matrix=coo_matrix((vv,(rr,cc)),shape=(len(lo),columns)).tocsc()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        solution=milp(costs,integrality=np.array([0]*z+[1]*len(candidates)),
                      bounds=Bounds(lower,upper),constraints=LinearConstraint(matrix,lo,hi),
                      options={'mip_rel_gap':0.0,'threads':1})
    assert solution.success and solution.mip_gap<1e-9,solution.message
    selected=[j for i,j in enumerate(candidates) if solution.x[z+i]>.5]
    assert len(selected)<=budget
    value,plans=menu_wait(problem,lead,selected)
    constant=problem['case']['delta']*sum(problem['weights'][s]*sum(problem['references'][s][t0:])
              for s in range(len(paths)))/problem['denominator']
    assert abs(solution.fun-constant-float(value))<1e-6
    assert np.max(np.abs(np.array(plans)-solution.x[:z].reshape(len(paths),h)))<1e-6
    return dict(budget=budget,waiting=float(value),waiting_exact=str(value),selected_slots=selected,
                method='message_binary_milp',gap=solution.mip_gap,nodes=int(solution.mip_node_count),
                candidate_messages=len(candidates),elapsed_seconds=perf_counter()-start)


def execute(smoke):
    old=design.read(OLD)
    rows=[]
    for row in old['rows']:
        if smoke and row['key'] not in old['smoke_keys']:
            continue
        path=ROOT/row['input_path']
        assert design.digest(path)==row['input_sha256']
        case=design.read(path)
        problem=design.prepare(case)
        if problem is None:
            continue
        lead=case['lag']+2
        minimum=design.menu_cover(problem,lead)['optimal_message_count']
        previous=None
        for budget in range(minimum+1):
            result=solve(problem,lead,budget)
            assert previous is None or result['waiting']<=previous+1e-8
            previous=result['waiting']
            result.update(case_id=row['case_id'],key=row['key'],capacity_label=row['capacity_label'],
                          N=case['n'],lead=lead,minimum_static_queries=minimum)
            rows.append(result)
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['smoke','panel'],required=True)
    args=parser.parse_args()
    start=perf_counter()
    if args.phase=='smoke':
        # Exhaustive menu oracle on the small complementary-information case.
        case=dict(n=1,caps=[1,0,0,1],lag=0,delta=1,
                  service=[[0,0,0,1],[0,1,0,0],[0,0,1,0]],probabilities=['1/3']*3)
        p=design.prepare(case)
        for budget in range(3):
            answer=solve(p,3,budget)
            oracle=min(menu_wait(p,3,slots)[0] for k in range(budget+1)
                       for slots in combinations(range(4),k))
            assert answer['waiting_exact']==str(oracle)
        rows=execute(True)
        report=dict(passed=True,source_sha256=design.digest(Path(__file__)),rows=rows,
                    elapsed_seconds=perf_counter()-start,max_seconds=max(r['elapsed_seconds'] for r in rows))
        design.save(OUT/'budget_smoke.json',report)
        print({k:v for k,v in report.items() if k!='rows'},flush=True)
    else:
        smoke=design.read(OUT/'budget_smoke.json')
        assert smoke['passed'] and smoke['source_sha256']==design.digest(Path(__file__))
        design.save(DATA/'budget_design.json',dict(date='2026-10-28',frozen_before_panel=True,
                    source_sha256=design.digest(Path(__file__)),input_sha256=design.digest(OLD),
                    smoke_sha256=design.digest(OUT/'budget_smoke.json'),prefix=0,lead='lag+2',
                    budgets='Every integer from zero to minimum static full-recovery menu size',
                    coverage='Every complete support among all 282 candidate configurations',
                    method='Messages binary; cumulative releases continuous; zero relative MIP gap; independent integer frontier replay.'))
        rows=execute(False)
        csv_write(OUT/'budget_frontiers.csv',[{k:v for k,v in row.items() if k!='selected_slots'} for row in rows])
        design.save(OUT/'budget_panel.json',dict(passed=True,rows=rows,elapsed_seconds=perf_counter()-start,
                    source_sha256=design.digest(Path(__file__))))
        print(dict(passed=True,frontier_points=len(rows),elapsed_seconds=perf_counter()-start),flush=True)


if __name__=='__main__':
    main()
