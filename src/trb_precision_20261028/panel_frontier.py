"""Compute every deterministic waiting-request trade-off on the frozen panel."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import os
from pathlib import Path
from time import perf_counter
from band_design import ROOT, BandDesign, StateLimit, old
from frontier_design import FrontierDesign
from panel_band import read, save, sha
from validate_band import replay

DATA=ROOT/'data/trb_precision_20261028'
OUT=ROOT/'results/trb_precision_20261028'

def one(task):
    from math import ceil
    import json
    job,d,w=task
    assert sha(ROOT/job['input_path'])==job['input_sha256']
    c=read(ROOT/job['input_path'])
    c=dict(c,caps=[ceil(c['n']/job['divisor'])]*len(c['caps']))
    p=old.prepare(c);lead=c['lag']+2
    row=dict(case_id=job['case_id'],capacity=job['capacity'],airport=job['airport'],
             day=job['day'],n=c['n'],delay=d,budget=2,width=w)
    try:
        model=FrontierDesign(p,lead,d,2,w,seconds_limit=60)
        result=model.solve();row.update(result)
        lex=BandDesign(p,lead,d,2,w).solve()
        first=row['frontier'][0]
        assert (first['weighted_waiting'],first['weighted_queries'])==(
                lex['weighted_waiting'],lex['weighted_queries'])
        assert row['frontier'][-1]['weighted_queries']==0
        prior=None
        out=OUT/'frontier_replay'/f"{job['case_id']}_{job['capacity']}_d{d}_w{w}.json.gz"
        out.parent.mkdir(parents=True,exist_ok=True)
        trajectories=0;histories=0
        with gzip.open(out,'wt',encoding='utf-8') as f:
            f.write('{"policies":[')
            for i,point in enumerate(row['frontier']):
                pair=(point['weighted_waiting'],point['weighted_queries'])
                if prior is not None:
                    assert pair[0]>prior[0] and pair[1]<prior[1]
                prior=pair
                paths=model.replay_pair(pair)
                histories+=replay(p,paths,lead,d,2,w)
                trajectories+=len(paths)
                if i: f.write(',')
                json.dump(dict(point=point,paths=paths),f)
            f.write(']}')
        row.update(trajectories_checked=trajectories,histories_checked=histories)
    except StateLimit as e:
        row.update(optimal=False,unresolved=str(e))
    return row

def smoke():
    assert read(OUT/'frontier_smoke.json')['passed']
    jobs=read(DATA/'panel_design.json')['jobs']
    selected=[next(j for j in jobs if j['airport']==a and j['capacity']=='N4')
              for a in ['DEN','PHX']]
    selected.append(next(j for j in jobs if j['case_id']=='1b72b60d2b7f_k1' and j['capacity']=='N'))
    rows=[one((j,d,w)) for j in selected for d in [1,2] for w in [1,2,4]]
    passed=all(r['optimal'] for r in rows)
    save(OUT/'frontier_public_smoke.json',dict(passed=passed,rows=rows))
    print(dict(passed=passed,conditions=len(rows),
               max_states=max(r['states'] for r in rows),
               max_frontier=max(len(r['frontier']) for r in rows)))
    assert passed

def prepare():
    assert read(OUT/'frontier_smoke.json')['passed'] and read(OUT/'frontier_public_smoke.json')['passed']
    names=['frontier_design.py','validate_frontier.py','panel_frontier.py',
           'band_design.py','unlimited_band.py','validate_band.py']
    design=dict(jobs=read(DATA/'panel_design.json')['jobs'],widths=[1,2,4],delays=[1,2],
                budget=2,workers=2,sources={n:sha(Path(__file__).parent/n) for n in names},
                evaluation='All nondominated deterministic policies; no selected monetary weight or tolerated waiting threshold.',
                matching='Every policy preserves all reference takeoffs; equal travel, service calendars, probabilities and limits.')
    save(DATA/'frontier_design.json',design)
    print(dict(conditions=len(design['jobs'])*6))

def run():
    design=read(DATA/'frontier_design.json')
    for n,digest in design['sources'].items(): assert sha(Path(__file__).parent/n)==digest
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    tasks=[(j,d,w) for j in design['jobs'] for d in design['delays'] for w in design['widths']]
    start=perf_counter();rows=[]
    with ProcessPoolExecutor(max_workers=design['workers']) as pool:
        for i,row in enumerate(pool.map(one,tasks),1):
            rows.append(row)
            if i%50==0:
                save(OUT/'frontier_progress.json',dict(conditions=i,seconds=perf_counter()-start))
                print(dict(conditions=i,unresolved=sum(not r['optimal'] for r in rows),
                           seconds=perf_counter()-start),flush=True)
    save(OUT/'frontier_rows.json',rows)
    save(OUT/'frontier_audit.json',dict(passed=all(r['optimal'] for r in rows),conditions=len(rows),
        policies=sum(len(r.get('frontier',[])) for r in rows),
        trajectories=sum(r.get('trajectories_checked',0) for r in rows),
        seconds=perf_counter()-start))
    print(read(OUT/'frontier_audit.json'))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['smoke','prepare','run'])
    globals()[parser.parse_args().phase]()
