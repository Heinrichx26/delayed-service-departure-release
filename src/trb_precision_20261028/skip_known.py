"""A fixed request timetable that omits only already determined replies."""
from collections import defaultdict
import argparse
import gzip
import json
from math import ceil
from statistics import mean
from time import perf_counter
from band_design import ROOT, old
from fixed_band import fixed_band
from panel_band import read, save, sha
from validate_band import replay

OUT=ROOT/'results/trb_precision_20261028'

def skip_known(problem, fixed_paths, lead, delay, budget, width):
    schedule=fixed_paths[0]['queries']
    assert all(r['queries']==schedule for r in fixed_paths)
    rows=[]
    for s,path in enumerate(problem['paths']):
        sent=[]
        release=list(problem['prefix'])
        previous=release[-1] if release else 0
        for t in range(problem['t0'],problem['horizon']):
            arrived=[q['slot'] for q in sent if q['arrival']<=t]
            members=[q for q,p in enumerate(problem['paths'])
                     if p[:t]==path[:t] and all(p[j]//width==path[j]//width for j in arrived)]
            assert s in members
            release.append(max(previous,max(problem['bounds'][q][t] for q in members)))
            for query in schedule:
                if query['dispatch']==t and len({
                        problem['paths'][q][query['slot']]//width for q in members})>1:
                    sent.append(query)
            previous=release[-1]
        assert release==fixed_paths[s]['release']
        done,inventory,_=old.base.physical(release,path,problem['n'],problem['lag'])
        assert done==problem['references'][s]
        rows.append(dict(s=s,queries=sent,release=release,completed=done,
            post_release=problem['case']['delta']*sum(inventory[problem['t0']:])/problem['n']))
    replay(problem,rows,lead,delay,budget,width)
    count=sum(problem['weights'][r['s']]*len(r['queries']) for r in rows)/problem['denominator']
    return count,rows

def smoke():
    import random
    rng=random.Random(2026102843);records=[]
    for i in range(12):
        n=3;paths=[]
        for _ in range(3):
            p=[0]*6
            for _ in range(n):p[rng.randrange(3,6)]+=1
            paths.append(p)
        case=dict(n=n,lag=1,caps=[1]*6,service=paths,
                  probabilities=['1/6','1/3','1/2'],delta=2)
        p=old.prepare(case)
        for d in [1,2]:
            for w in [1,2,4]:
                fixed=fixed_band(p,3,d,2,w)
                count,rows=skip_known(p,fixed['paths'],3,d,2,w)
                assert count<=fixed['count']+1e-10
                records.append(dict(case=i,delay=d,width=w,fixed=fixed['count'],omission=count))
    jobs=read(ROOT/'data/trb_precision_20261028/panel_design.json')['jobs']
    for airport in ['DEN','PHX']:
        job=next(j for j in jobs if j['airport']==airport and j['capacity']=='N4')
        c=read(ROOT/job['input_path'])
        c=dict(c,caps=[ceil(c['n']/job['divisor'])]*len(c['caps']))
        p=old.prepare(c)
        for d in [1,2]:
            for w in [1,2,4]:
                if w==1:
                    path=ROOT/'results/trb_joint_20261028/fixed_replay'/f"{job['case_id']}_{job['capacity']}_d{d}_b2.json.gz"
                    with gzip.open(path,'rt',encoding='utf-8') as f:data=json.load(f)
                    paths=data['paths'];fixed=data['result']['count']
                else:
                    path=OUT/'replay'/f"{job['case_id']}_{job['capacity']}_d{d}_b2_w{w}.json.gz"
                    with gzip.open(path,'rt',encoding='utf-8') as f:data=json.load(f)
                    paths=data['fixed'];fixed=data['result']['fixed']['count']
                count,rows=skip_known(p,paths,c['lag']+2,d,2,w)
                assert count<=fixed
                records.append(dict(airport=airport,delay=d,width=w,fixed=fixed,omission=count))
    save(OUT/'skip_smoke.json',dict(passed=True,conditions=len(records),rows=records))
    print(dict(passed=True,conditions=len(records)))

def run():
    assert read(OUT/'skip_smoke.json')['passed']
    jobs=read(ROOT/'data/trb_precision_20261028/panel_design.json')['jobs']
    design=dict(jobs=jobs,delays=[1,2],widths=[1,2,4],budget=2,source_sha256=sha(__import__('pathlib').Path(__file__)),
                rule='Retain a precomputed optimal fixed timetable; omit dispatch only when its response is constant across the currently compatible calendars.',
                tie='Use the fixed optimizer solution already saved, without optimizing which tied timetable benefits most from later omission.')
    save(ROOT/'data/trb_precision_20261028/skip_design.json',design)
    start=perf_counter();rows=[];paths_checked=0
    for job in jobs:
        c=read(ROOT/job['input_path']);c=dict(c,caps=[ceil(c['n']/job['divisor'])]*len(c['caps']))
        p=old.prepare(c)
        for d in [1,2]:
            for w in [1,2,4]:
                if w==1:
                    path=ROOT/'results/trb_joint_20261028/fixed_replay'/f"{job['case_id']}_{job['capacity']}_d{d}_b2.json.gz"
                    with gzip.open(path,'rt',encoding='utf-8') as f: data=json.load(f)
                    fixed_paths=data['paths'];fx=data['result']['count']
                    adapt=next(r for r in read(ROOT/'results/trb_joint_20261028/rows.json')
                         if r['case_id']==job['case_id'] and r['capacity']==job['capacity'] and r['delay']==d and r['budget']==2)['expected_queries']
                else:
                    path=OUT/'replay'/f"{job['case_id']}_{job['capacity']}_d{d}_b2_w{w}.json.gz"
                    with gzip.open(path,'rt',encoding='utf-8') as f: data=json.load(f)
                    fixed_paths=data['fixed'];fx=data['result']['fixed']['count']
                    adapt=data['result']['adaptive']['expected_queries']
                count,paths=skip_known(p,fixed_paths,c['lag']+2,d,2,w)
                assert adapt<=count+1e-10<=fx+1e-10
                paths_checked+=len(paths)
                row=dict(case_id=job['case_id'],capacity=job['capacity'],delay=d,width=w,
                         fixed=fx,omission=count,optimal=adapt)
                rows.append(row)
                output=OUT/'skip_replay'/f"{job['case_id']}_{job['capacity']}_d{d}_w{w}.json.gz"
                output.parent.mkdir(parents=True,exist_ok=True)
                with gzip.open(output,'wt',encoding='utf-8') as f:json.dump(dict(result=row,paths=paths),f)
    summary=[]
    for d in [1,2]:
        for w in [1,2,4]:
            g=[r for r in rows if r['delay']==d and r['width']==w]
            summary.append(dict(delay=d,width=w,conditions=len(g),
                fixed=mean(r['fixed'] for r in g),omission=mean(r['omission'] for r in g),
                optimal=mean(r['optimal'] for r in g),
                residual=mean(r['omission']-r['optimal'] for r in g),
                improved=sum(r['omission']>r['optimal']+1e-10 for r in g),
                max_residual=max(r['omission']-r['optimal'] for r in g)))
    save(OUT/'skip_rows.json',rows);save(OUT/'skip_summary.json',summary)
    save(OUT/'skip_audit.json',dict(passed=True,conditions=len(rows),trajectories=paths_checked,seconds=perf_counter()-start))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['smoke','run'])
    globals()[parser.parse_args().phase]()
