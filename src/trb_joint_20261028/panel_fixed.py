"""Fixed-schedule ablation, with enumeration before public-data execution."""
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import argparse
import gzip
import json
from math import ceil
import os
from pathlib import Path
import random
from statistics import mean
from time import perf_counter
from joint_design import ROOT,JointDesign,old
from fixed_joint import fixed_joint
from validate_joint import replay
from panel_joint import read,save,sha

OUT=ROOT/'results/trb_joint_20261028'

def brute(problem,lead,delay,budget):
    h=problem['horizon'];paths=problem['paths'];m=len(paths)
    t0=problem['t0']
    options=[(r,j) for r in range(t0,h-delay) for j in range(r+delay,min(h,r+lead+1))
             if len({p[j] for p in paths})>1]
    best=None
    for size in range(min(h,len(options))+1):
        for selected in combinations(options,size):
            if len({j for _,j in selected})<size:continue
            if any(sum(r==t for r,j in selected)>budget for t in range(h)):continue
            releases=[list(problem['prefix']) for _ in paths]
            for t in range(t0,h):
                known=list(range(t))+[j for r,j in selected if r+delay<=t]
                for s,path in enumerate(paths):
                    members=[q for q,p in enumerate(paths) if all(path[j]==p[j] for j in known)]
                    previous=releases[s][-1] if t else 0
                    releases[s].append(max(previous,max(problem['bounds'][q][t] for q in members)))
            weighted=sum(w*sum(u-d for u,d in zip(releases[s][t0:],problem['references'][s][t0:]))
                         for s,w in enumerate(problem['weights']))
            pair=weighted,size
            if best is None or pair<best:best=pair
    return best

def smoke():
    start=perf_counter();rng=random.Random(2026102819);rows=[]
    for i in range(12):
        n=rng.choice([1,2]);m=3;paths=[]
        for _ in range(m):
            p=[0]*6
            for k in range(n):p[rng.randrange(2,6)]+=1
            paths.append(p)
        case=dict(n=n,lag=1,caps=[1]*6,service=paths,probabilities=['1/3']*m,delta=2)
        problem=old.prepare(case)
        for d in [1,2]:
            for b in [1,2]:
                answer=fixed_joint(problem,3,d,b);oracle=brute(problem,3,d,b)
                assert answer['optimal'] and (answer['weighted_waiting'],answer['count'])==oracle
                replay(problem,answer['paths'],3,d,b)
                adaptive=JointDesign(problem,3,d,b).solve()
                assert adaptive['waiting']<=answer['waiting']+1e-9
                rows.append(dict(case=i,delay=d,budget=b,value=oracle))
        reset=old.prepare(case,prefix=old.base.earliest(n,case['caps'])[:2])
        for d in [1,2]:
            for b in [1,2]:
                answer=fixed_joint(reset,3,d,b);oracle=brute(reset,3,d,b)
                assert answer['optimal'] and (answer['weighted_waiting'],answer['count'])==oracle
                replay(reset,answer['paths'],3,d,b)
                rows.append(dict(case=i,reset=2,delay=d,budget=b,value=oracle))
    jobs=read(ROOT/'data/trb_latency_20261028/panel_design.json')['jobs']
    public=[]
    for airport in ['DEN','PHX']:
        job=next(j for j in jobs if j['airport']==airport and j['capacity']=='N4')
        for d in [1,2]:
            for b in [1,2]:public.append(one((job,d,b)))
    assert all(r['optimal'] for r in public)
    save(OUT/'fixed_smoke.json',dict(passed=True,comparisons=len(rows),rows=rows,public=public,seconds=perf_counter()-start))
    print(json.dumps(dict(passed=True,enumerations=len(rows),public=len(public),seconds=perf_counter()-start)))

def one(task):
    j,d,b=task
    assert sha(ROOT/j['input_path'])==j['input_sha256']
    c=read(ROOT/j['input_path']);c=dict(c,caps=[ceil(c['n']/j['divisor'])]*len(c['caps']))
    p=old.prepare(c)
    answer=fixed_joint(p,c['lag']+2,d,b)
    row=dict(case_id=j['case_id'],capacity=j['capacity'],airport=j['airport'],day=j['day'],delay=d,budget=b,
             **{k:v for k,v in answer.items() if k not in ['paths','schedule']})
    if answer['optimal']:
        replay(p,answer['paths'],c['lag']+2,d,b)
        out=OUT/'fixed_replay'/f"{j['case_id']}_{j['capacity']}_d{d}_b{b}.json.gz"
        out.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(out,'wt',encoding='utf-8') as f:json.dump(dict(job=j,result=row,paths=answer['paths']),f)
    return row

def run():
    assert read(OUT/'fixed_smoke.json')['passed']
    jobs=read(ROOT/'data/trb_joint_20261028/panel_design.json')['jobs']
    design=dict(jobs=jobs,source_sha256=sha(Path(__file__)),model_sha256=sha(Path(__file__).with_name('fixed_joint.py')),workers=2)
    save(ROOT/'data/trb_joint_20261028/fixed_design.json',design)
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    start=perf_counter();rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,row in enumerate(pool.map(one,[(j,d,b) for j in jobs for d in [1,2] for b in [1,2]]),1):
            rows.append(row)
            if i%25==0:
                save(OUT/'fixed_progress.json',dict(rows=rows,seconds=perf_counter()-start))
                print(json.dumps(dict(conditions=i,unresolved=sum(not r['optimal'] for r in rows),seconds=perf_counter()-start)),flush=True)
    save(OUT/'fixed_rows.json',rows)
    key=lambda r:(r['case_id'],r['capacity'],r['delay'],r['budget'])
    adapt={key(r):r for r in read(OUT/'rows.json')}
    summary=[]
    for d in [1,2]:
        for b in [1,2]:
            group=[r for r in rows if r['delay']==d and r['budget']==b and r['optimal']]
            for r in group:
                a=adapt[key(r)]
                assert a['weighted_waiting']<=r['weighted_waiting']
                if a['weighted_waiting']==r['weighted_waiting']:assert a['expected_queries']<=r['count']+1e-9
            summary.append(dict(delay=d,budget=b,solved=len(group),fixed_waiting=mean(r['waiting'] for r in group),
                adaptive_waiting=mean(adapt[key(r)]['waiting'] for r in group),
                adaptive_saving=mean(r['waiting']-adapt[key(r)]['waiting'] for r in group),
                strictly_improved=sum(adapt[key(r)]['waiting']<r['waiting']-1e-9 for r in group),
                fixed_queries=mean(r['count'] for r in group),
                adaptive_queries=mean(adapt[key(r)]['expected_queries'] for r in group)))
    save(OUT/'fixed_summary.json',summary)
    save(OUT/'fixed_audit.json',dict(passed=all(r['optimal'] for r in rows),conditions=len(rows),seconds=perf_counter()-start,summary=summary))
    print(json.dumps(summary))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['smoke','run']);a=p.parse_args();globals()[a.phase]()
