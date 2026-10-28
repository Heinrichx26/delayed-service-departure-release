"""Audit attainable alternatives and exact minimum per-slot request capacity."""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import json
from math import ceil
from pathlib import Path
from statistics import mean
from time import perf_counter
from delayed_acquisition import ROOT,DelayedDesign,delayed_targets,fixed_schedule,old
from validate import independent_replay

OUT=ROOT/'results/trb_latency_20261028'
DATA=ROOT/'data/trb_latency_20261028'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')

def run(smoke=False):
    started=perf_counter();design=read(DATA/'panel_design.json');prior=read(OUT/'rows.json')
    source=Path(__file__);protocol=DATA/'capacity_design.json'
    if smoke:
        save(protocol,dict(source_sha256=sha(source),parent_rows_sha256=sha(OUT/'rows.json'),
             parent_design_sha256=sha(DATA/'panel_design.json'),
             rule='Search integer request limits upward for the same delayed target; reuse certified b=1,2 answers. Export independently replayed attainable policies.',
             upper_bound='L-delay+1 nonredundant requests in one slot',
             no_new_operational_assumptions=True))
    else:
        p=read(protocol);assert p['source_sha256']==sha(source)
        assert p['parent_rows_sha256']==sha(OUT/'rows.json')
        assert read(OUT/'capacity_smoke.json')['passed']
        assert (OUT/'capacity_smoke.json').stat().st_mtime<=(OUT/'audit.json').stat().st_mtime or read(OUT/'audit.json')['passed']
    rows=[];replayed=0;strict=0
    jobs=design['jobs']
    if smoke:
        difficult=next(r for r in prior if r['budget']==2 and not r['fixed']['feasible'])
        jobs=[jobs[0],next(j for j in jobs if j['case_id']==difficult['case_id'] and j['capacity']==difficult['capacity'])]
    for job in jobs:
        case=read(ROOT/job['input_path']);assert sha(ROOT/job['input_path'])==job['input_sha256']
        case=dict(case,caps=[ceil(case['n']/job['divisor'])]*len(case['caps']))
        problem=old.prepare(case);lead=case['lag']+2
        for delay in [1,2]:
            target=delayed_targets(problem,lead,delay)
            known=[r for r in prior if r['case_id']==job['case_id'] and r['capacity']==job['capacity'] and r['delay']==delay]
            adaptive_min=next((r['budget'] for r in known if r['adaptive']['feasible']),None)
            fixed_min=next((r['budget'] for r in known if r['fixed']['feasible']),None)
            solved=[]
            for budget in range(3,lead-delay+2):
                if fixed_min is not None and adaptive_min is not None:break
                if adaptive_min is None:
                    model=DelayedDesign(problem,lead,delay,budget,target)
                    a=model.solve()
                    if a['feasible']:
                        replay=model.replay();assert independent_replay(problem,target,replay,lead,delay,budget)==a['weighted_cost']
                        adaptive_min=budget
                        folder=OUT/'capacity_replay';folder.mkdir(exist_ok=True)
                        with gzip.open(folder/f"{job['case_id']}_{job['capacity']}_d{delay}_b{budget}.json.gz",'wt',encoding='utf-8') as f:
                            json.dump(dict(job=job,target=target,answer=a,replay=replay),f)
                    else:replay=[]
                else:a=None
                if fixed_min is None:
                    f=fixed_schedule(problem,lead,delay,budget,target);assert f['optimal']
                    if f['feasible']:fixed_min=budget
                else:f=None
                solved.append(dict(budget=budget,adaptive=a,fixed=f))
            assert adaptive_min is not None and fixed_min is not None
            assert adaptive_min<=fixed_min
            strict+=adaptive_min<fixed_min
            rows.append(dict(case_id=job['case_id'],airport=job['airport'],day=job['day'],capacity=job['capacity'],
                             delay=delay,adaptive_min=adaptive_min,fixed_min=fixed_min,extra_solves=solved))
            for r in known:
                budget=r['budget'];alt=r['attainable_adaptive'];alt_target=delayed_targets(problem,alt['target_lead'],delay)
                model=DelayedDesign(problem,lead,delay,budget,alt_target);answer=model.solve()
                assert answer['feasible'] and answer['weighted_cost']==alt['weighted_cost']
                replay=model.replay();assert independent_replay(problem,alt_target,replay,lead,delay,budget)==answer['weighted_cost']
                fixed=r['attainable_fixed'];ftarget=delayed_targets(problem,fixed['target_lead'],delay)
                fixed_replay=[]
                for s,path in enumerate(problem['paths']):
                    release=ftarget[s];done,inv,_=old.base.physical(release,path,problem['n'],problem['lag'])
                    fixed_replay.append(dict(s=s,queries=fixed['schedule'],release=release,completed=done,post_release=case['delta']*sum(inv)/case['n']))
                assert independent_replay(problem,ftarget,fixed_replay,lead,delay,budget)==fixed['count']*problem['denominator']
                folder=OUT/('attainable_smoke_replay' if smoke else 'attainable_replay');folder.mkdir(exist_ok=True)
                with gzip.open(folder/f"{job['case_id']}_{job['capacity']}_d{delay}_b{budget}.json.gz",'wt',encoding='utf-8') as f:
                    json.dump(dict(job=job,delay=delay,budget=budget,adaptive_target=alt_target,adaptive_replay=replay,
                                   fixed_target=ftarget,fixed_replay=fixed_replay),f)
                replayed+=1
    summary=[]
    for delay in [1,2]:
        group=[r for r in rows if r['delay']==delay]
        summary.append(dict(delay=delay,cases=len(group),adaptive_distribution=dict(sorted(Counter(r['adaptive_min'] for r in group).items())),
                            fixed_distribution=dict(sorted(Counter(r['fixed_min'] for r in group).items())),
                            adaptive_mean=mean(r['adaptive_min'] for r in group),fixed_mean=mean(r['fixed_min'] for r in group),
                            adaptive_max=max(r['adaptive_min'] for r in group),fixed_max=max(r['fixed_min'] for r in group)))
    report=dict(passed=True,smoke=smoke,minimum_capacity_cases=len(rows),attainable_policies_replayed=replayed,
                adaptive_strict_capacity_gain=strict,summary=summary,source_sha256=sha(source),
                design_sha256=sha(protocol),seconds=perf_counter()-started)
    suffix='smoke' if smoke else 'audit'
    save(OUT/f'capacity_{suffix}.json',report)
    save(OUT/f'capacity_{suffix}_rows.json',rows)
    print(json.dumps(report))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['smoke','run']);args=parser.parse_args()
    run(args.phase=='smoke')
