"""Evaluate joint design on exactly the earlier complete-case panel."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from math import ceil
import os
from statistics import mean
from time import perf_counter
from joint_design import JointDesign,StateLimit,ROOT,old
from validate_joint import replay

OUT=ROOT/'results/trb_joint_20261028'
DATA=ROOT/'data/trb_joint_20261028'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def one(item):
    job,delay,budget=item
    assert sha(ROOT/job['input_path'])==job['input_sha256']
    case=read(ROOT/job['input_path']);case=dict(case,caps=[ceil(case['n']/job['divisor'])]*len(case['caps']))
    problem=old.prepare(case);assert problem is not None
    row=dict(case_id=job['case_id'],airport=job['airport'],day=job['day'],capacity=job['capacity'],
             delay=delay,budget=budget,n=case['n'],lead=case['lag']+2)
    try:
        model=JointDesign(problem,row['lead'],delay,budget)
        row.update(model.solve());paths=model.replay()
        row['information_states_checked']=replay(problem,paths,row['lead'],delay,budget)
        target=OUT/'replay'/f"{job['case_id']}_{job['capacity']}_d{delay}_b{budget}.json.gz"
        target.parent.mkdir(exist_ok=True,parents=True)
        with gzip.open(target,'wt',encoding='utf-8') as f:json.dump(dict(job=job,result=row,paths=paths),f)
    except StateLimit as e:row.update(optimal=False,unresolved=str(e))
    return row

def smoke():
    assert read(OUT/'theory_smoke.json')['passed']
    jobs=read(ROOT/'data/trb_latency_20261028/panel_design.json')['jobs']
    selected=[next(j for j in jobs if j['airport']==a and j['capacity']=='N4') for a in ['DEN','PHX']]
    rows=[]
    for j in selected:
        for d,b in [(1,1),(1,2),(2,1),(2,2)]:
            row=one((j,d,b));rows.append(row);print(json.dumps(row),flush=True)
    passed=all(r['optimal'] for r in rows)
    result=dict(passed=passed,rows=rows,model_sha256=sha(Path(__file__).with_name('joint_design.py')))
    save(OUT/'public_smoke.json',result)
    assert passed

def prepare():
    assert read(OUT/'theory_smoke.json')['passed'] and read(OUT/'public_smoke.json')['passed']
    from pathlib import Path
    source=Path(__file__)
    parent=ROOT/'data/trb_latency_20261028/panel_design.json'
    jobs=read(parent)['jobs']
    design=dict(parent_sha256=sha(parent),jobs=jobs,delays=[1,2],budgets=[1,2],workers=2,
                source_sha256=sha(source),model_sha256=sha(source.with_name('joint_design.py')),
                validation_sha256=sha(source.with_name('validate_joint.py')))
    save(DATA/'panel_design.json',design);print(json.dumps(dict(jobs=len(jobs),conditions=len(jobs)*4)))

def run():
    from pathlib import Path
    design=read(DATA/'panel_design.json')
    assert sha(Path(__file__))==design['source_sha256']
    assert sha(Path(__file__).with_name('joint_design.py'))==design['model_sha256']
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    start=perf_counter();rows=[]
    tasks=[(j,d,b) for j in design['jobs'] for d in design['delays'] for b in design['budgets']]
    with ProcessPoolExecutor(max_workers=design['workers']) as pool:
        for i,row in enumerate(pool.map(one,tasks),1):
            rows.append(row)
            if i%25==0:
                save(OUT/'progress.json',dict(rows=rows,seconds=perf_counter()-start))
                print(json.dumps(dict(conditions=i,unresolved=sum(not r['optimal'] for r in rows),seconds=perf_counter()-start)),flush=True)
    save(OUT/'rows.json',rows)
    previous=read(ROOT/'results/trb_latency_20261028/rows.json')
    keys=lambda r:(r['case_id'],r['capacity'],r['delay'],r['budget'])
    before={keys(r):r for r in previous};summary=[]
    for d in [1,2]:
        for b in [1,2]:
            group=[r for r in rows if r['delay']==d and r['budget']==b and r['optimal']]
            for r in group:
                p=before[keys(r)]
                assert r['waiting']<=p['attainable_adaptive']['waiting']+1e-9
                assert r['waiting']>=p['delayed_target_waiting']-1e-9
            summary.append(dict(delay=d,budget=b,solved=len(group),
                improved=sum(r['waiting']<before[keys(r)]['attainable_adaptive']['waiting']-1e-9 for r in group),
                uniform_waiting=mean(before[keys(r)]['attainable_adaptive']['waiting'] for r in group),
                joint_waiting=mean(r['waiting'] for r in group),
                mean_saving=mean(before[keys(r)]['attainable_adaptive']['waiting']-r['waiting'] for r in group),
                max_saving=max(before[keys(r)]['attainable_adaptive']['waiting']-r['waiting'] for r in group),
                excess_over_unlimited=mean(r['waiting']-before[keys(r)]['delayed_target_waiting'] for r in group),
                queries=mean(r['expected_queries'] for r in group),
                max_states=max(r['states'] for r in group)))
    save(OUT/'summary.json',summary);save(OUT/'audit.json',dict(passed=all(r['optimal'] for r in rows),conditions=len(rows),seconds=perf_counter()-start,summary=summary))
    print(json.dumps(summary))

from pathlib import Path
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['smoke','prepare','run']);a=p.parse_args()
    globals()[a.phase]()
