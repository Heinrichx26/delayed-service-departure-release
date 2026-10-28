"""Optimize the omission baseline over all admissible nominal timetables."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from math import ceil
import os
from pathlib import Path
from statistics import mean
from time import perf_counter
from band_design import ROOT,old
from nominal_omission import optimize_omission
from panel_band import read,save,sha
from validate_band import replay

OUT=ROOT/'results/trb_precision_20261028'
DATA=ROOT/'data/trb_precision_20261028'

def one(task):
    job,d,w,minimum,expected,unoptimized=task
    assert sha(ROOT/job['input_path'])==job['input_sha256']
    c=read(ROOT/job['input_path']);c=dict(c,caps=[ceil(c['n']/job['divisor'])]*len(c['caps']))
    p=old.prepare(c)
    answer=optimize_omission(p,c['lag']+2,d,2,w,known_waiting=minimum,seconds=25)
    row=dict(case_id=job['case_id'],capacity=job['capacity'],airport=job['airport'],
             day=job['day'],delay=d,width=w,budget=2,adaptive=expected,
             arbitrary_fixed_omission=unoptimized,
             **{k:v for k,v in answer.items() if k not in ['paths','schedule']})
    if not answer['optimal']:return row
    assert expected<=answer['expected_queries']+1e-9<=unoptimized+1e-9
    assert minimum==answer['weighted_waiting']
    replay(p,answer['paths'],c['lag']+2,d,2,w)
    out=OUT/'optimized_omission_replay'/f"{job['case_id']}_{job['capacity']}_d{d}_w{w}.json.gz"
    out.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(out,'wt',encoding='utf-8') as f:
        json.dump(dict(result=row,nominal_schedule=answer['schedule'],paths=answer['paths']),f)
    return row

def tasks(jobs):
    frontier={(r['case_id'],r['capacity'],r['delay'],r['width']):r
              for r in read(OUT/'frontier_rows.json')}
    omission={(r['case_id'],r['capacity'],r['delay'],r['width']):r
              for r in read(OUT/'skip_rows.json')}
    return [(j,d,w,frontier[(j['case_id'],j['capacity'],d,w)]['frontier'][0]['weighted_waiting'],
             frontier[(j['case_id'],j['capacity'],d,w)]['frontier'][0]['expected_queries'],
             omission[(j['case_id'],j['capacity'],d,w)]['omission'])
            for j in jobs for d in [1,2] for w in [1,2,4]]

def smoke():
    assert read(OUT/'omission_theory_smoke.json')['passed']
    jobs=read(DATA/'panel_design.json')['jobs']
    chosen=[next(j for j in jobs if j['airport']==a and j['capacity']=='N4') for a in ['DEN','PHX']]
    chosen.append(next(j for j in jobs if j['case_id']=='1b72b60d2b7f_k1' and j['capacity']=='N'))
    start=perf_counter();rows=[one(t) for t in tasks(chosen)]
    passed=all(r['optimal'] for r in rows)
    save(OUT/'omission_public_smoke.json',dict(passed=passed,conditions=len(rows),rows=rows,seconds=perf_counter()-start))
    print(json.dumps(dict(passed=passed,conditions=len(rows),seconds=perf_counter()-start,
        max_variables=max(r.get('variables',0) for r in rows),
        max_case_seconds=max(r.get('seconds',0) for r in rows))))
    assert passed

def prepare():
    assert read(OUT/'omission_theory_smoke.json')['passed'] and read(OUT/'omission_public_smoke.json')['passed']
    names=['nominal_omission.py','skip_known.py','panel_omission.py','validate_omission.py']
    save(DATA/'optimized_omission_design.json',dict(jobs=read(DATA/'panel_design.json')['jobs'],
         delays=[1,2],widths=[1,2,4],budget=2,workers=2,
         sources={n:sha(Path(__file__).parent/n) for n in names},
         objective='Minimize probability-weighted executed checks over all nominal timetables attaining the proven minimum waiting.',
         capacity='The nominal timetable has at most two planned requests per slot; determined responses are omitted at dispatch.'))
    print('Prepared 1200 conditions with source hashes.')

def run():
    design=read(DATA/'optimized_omission_design.json')
    for n,digest in design['sources'].items():assert sha(Path(__file__).parent/n)==digest
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    start=perf_counter();rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,row in enumerate(pool.map(one,tasks(design['jobs'])),1):
            rows.append(row)
            if i%50==0:
                save(OUT/'omission_progress.json',dict(conditions=i,rows=rows,seconds=perf_counter()-start))
                print(dict(conditions=i,unresolved=sum(not r['optimal'] for r in rows),
                           seconds=perf_counter()-start),flush=True)
    save(OUT/'optimized_omission_rows.json',rows)
    summary=[]
    for d in [1,2]:
        for w in [1,2,4]:
            g=[r for r in rows if r['delay']==d and r['width']==w and r['optimal']]
            summary.append(dict(delay=d,width=w,solved=len(g),
                executed=mean(r['expected_queries'] for r in g),
                adaptive=mean(r['adaptive'] for r in g),
                residual=mean(r['expected_queries']-r['adaptive'] for r in g),
                improved=sum(r['expected_queries']>r['adaptive']+1e-9 for r in g),
                max_residual=max(r['expected_queries']-r['adaptive'] for r in g)))
    save(OUT/'optimized_omission_summary.json',summary)
    save(OUT/'optimized_omission_audit.json',dict(passed=all(r['optimal'] for r in rows),
         conditions=len(rows),trajectories=sum(r['optimal'] for r in rows)*16,
         seconds=perf_counter()-start))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['smoke','prepare','run'])
    globals()[parser.parse_args().phase]()
