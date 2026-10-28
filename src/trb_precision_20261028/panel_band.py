"""Frozen same-input precision comparison after exhaustive and public smoke."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from math import ceil
import os
from pathlib import Path
from statistics import mean
from time import perf_counter
from band_design import BandDesign, StateLimit, ROOT, old
from fixed_band import fixed_band
from validate_band import replay
from unlimited_band import unlimited

DATA = ROOT/'data/trb_precision_20261028'
OUT = ROOT/'results/trb_precision_20261028'
def read(path): return json.loads(path.read_text(encoding='utf-8'))
def save(path,value): path.parent.mkdir(exist_ok=True,parents=True);path.write_text(json.dumps(value,indent=2),encoding='utf-8')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def one(task):
    job,delay,budget,width = task
    assert sha(ROOT/job['input_path']) == job['input_sha256']
    c = read(ROOT/job['input_path'])
    c = dict(c,caps=[ceil(c['n']/job['divisor'])]*len(c['caps']))
    p = old.prepare(c)
    row = dict(case_id=job['case_id'],capacity=job['capacity'],airport=job['airport'],
               day=job['day'],n=c['n'],delay=delay,budget=budget,width=width)
    try:
        model = BandDesign(p,c['lag']+2,delay,budget,width)
        adaptive = model.solve()
        trajectories = model.replay()
        row['histories_checked'] = replay(p,trajectories,c['lag']+2,delay,budget,width)
        fixed = fixed_band(p,c['lag']+2,delay,budget,width)
        row['adaptive'] = adaptive
        row['fixed'] = {k:v for k,v in fixed.items() if k != 'paths'}
        row['optimal'] = adaptive['optimal'] and fixed['optimal']
        if not row['optimal']: return row
        replay(p,fixed['paths'],c['lag']+2,delay,budget,width)
        floor = unlimited(p,c['lag']+2,delay,width)
        exact_floor = unlimited(p,c['lag']+2,delay,1)
        row['unrestricted_waiting'] = floor['waiting']
        row['exact_unrestricted_waiting'] = exact_floor['waiting']
        assert adaptive['weighted_waiting'] >= floor['weighted_waiting'] >= exact_floor['weighted_waiting']
        assert fixed['weighted_waiting'] >= adaptive['weighted_waiting']
        if fixed['weighted_waiting'] == adaptive['weighted_waiting']:
            assert fixed['count'] >= adaptive['expected_queries']-1e-9
        # Sending every admissible report has at most lead-delay+1 new reports per slot.
        # At this capacity, the joint solver must attain the independently computed floor.
        if budget >= c['lag']+2-delay+1:
            assert adaptive['weighted_waiting'] == floor['weighted_waiting']
        out = OUT/'replay'/f"{job['case_id']}_{job['capacity']}_d{delay}_b{budget}_w{width}.json.gz"
        out.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(out,'wt',encoding='utf-8') as f:
            json.dump(dict(job=job,result=row,adaptive=trajectories,
                           fixed=fixed['paths'],unrestricted=floor['paths']),f)
    except StateLimit as e:
        row.update(optimal=False,unresolved=str(e))
    return row

def smoke():
    assert read(OUT/'theory_smoke.json')['passed']
    jobs = read(ROOT/'data/trb_latency_20261028/panel_design.json')['jobs']
    selected = [next(j for j in jobs if j['airport']==a and j['capacity']=='N4') for a in ['DEN','PHX']]
    rows = []
    for job in selected:
        for d,b in [(1,1),(1,2),(2,1),(2,2)]:
            for w in [1,2,4]:
                row=one((job,d,b,w)); rows.append(row)
    passed = all(r['optimal'] for r in rows)
    save(OUT/'public_smoke.json',dict(passed=passed,rows=rows))
    print(json.dumps(dict(passed=passed,conditions=len(rows),
                         max_states=max(r['adaptive']['states'] for r in rows))))
    assert passed

def prepare():
    assert read(OUT/'theory_smoke.json')['passed'] and read(OUT/'fixed_smoke.json')['passed']
    assert read(OUT/'public_smoke.json')['passed']
    jobs = read(ROOT/'data/trb_latency_20261028/panel_design.json')['jobs']
    sources = {p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    design = dict(jobs=jobs,widths=[2,4],delays=[1,2],budgets=[1,2],workers=2,
                  sources=sources,baseline='results/trb_joint_20261028/rows.json',
                  rationale=dict(widths='Nested partitions of one, two and four integer counts, matching the existing immediate-report study; sensitivity levels, not empirical response-error estimates.',
                  matching='All calendars, probabilities, release limits, travel times and takeoffs are unchanged.',
                  exclusion='Use all 200 previously frozen complete combinations; retain prior incomplete-combination analyses separately.'),
                  outcomes=['minimum waiting','minimum expected requests at that waiting',
                            'unrestricted-request lower bound','fixed-schedule optimum'])
    save(DATA/'panel_design.json',design)
    print(json.dumps(dict(conditions=len(jobs)*8,source_files=len(sources))))

def run():
    design = read(DATA/'panel_design.json')
    for name,digest in design['sources'].items(): assert sha(Path(__file__).parent/name)==digest
    os.environ['OMP_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'
    tasks=[(j,d,b,w) for j in design['jobs'] for d in design['delays']
           for b in design['budgets'] for w in design['widths']]
    rows=[];start=perf_counter()
    with ProcessPoolExecutor(max_workers=design['workers']) as pool:
        for i,row in enumerate(pool.map(one,tasks),1):
            rows.append(row)
            if i%50==0:
                save(OUT/'progress.json',dict(conditions=i,rows=rows,seconds=perf_counter()-start))
                print(json.dumps(dict(conditions=i,unresolved=sum(not r['optimal'] for r in rows),
                                      seconds=perf_counter()-start)),flush=True)
    save(OUT/'rows.json',rows)
    save(OUT/'audit.json',dict(passed=all(r['optimal'] for r in rows),
                             conditions=len(rows),seconds=perf_counter()-start))
    print(json.dumps(read(OUT/'audit.json')))

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['smoke','prepare','run'])
    globals()[parser.parse_args().phase]()
