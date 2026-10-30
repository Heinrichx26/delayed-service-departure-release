"""Certify a saved feasible policy with finite-depth admissible lower bounds."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import argparse
import hashlib
import json
from math import ceil
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/trb_error_20261030'))
from robust_reports import prepare,SearchLimit
from budgeted_reports import BudgetedReports

OUT=ROOT/'results/trb_calibrated_20261030'

def one(pair):
    r,policy=pair
    path=ROOT/Path(r['input_path'].replace('\\','/'))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==r['input_sha256']
    c=json.loads(path.read_text());divisor={'N':1,'N4':4,'N8':8}[r['capacity']]
    p=prepare(c['service'],c['n'],(ceil(c['n']/divisor),)*len(c['caps']),c['lag'])
    cap=policy['lookahead']['max_requests'];upper=policy['lookahead']['value']
    model=BudgetedReports(p,c['lag']+2,r['delay'],1,1,total_requests=cap,
                          seconds_limit=20,state_limit=500000)
    start=perf_counter()
    @lru_cache(None)
    def lower(depth,t,e,pending,queried,u):
        if t==model.h:return 0
        if depth==0:return model.lower(t,e,u)
        if perf_counter()-start>20 or lower.cache_info().misses>500000:
            raise SearchLimit('certificate resource allowance reached')
        z=model.release(t,e,u)
        if z==model.n:return model.lower(t,e,u)
        best=None
        for batch in model.batches(t,e,pending,queried):
            children=model.transitions(t,e,pending,queried,batch)
            cheap=max(z-ref+model.lower(t+1,en,z) for _,_,en,_,_,ref in children)
            if best is not None and cheap>=best:continue
            worst=0
            for _,_,en,pn,an,ref in children:
                worst=max(worst,z-ref+lower(depth-1,t+1,en,pn,an,z))
                if best is not None and worst>=best:break
            if best is None or worst<best:best=worst
        return best
    row={k:r[k] for k in ('case_id','airport','day','capacity','delay','n')}
    row.update(upper=upper,total_request_limit=cap,depths=[])
    try:
        previous=model.lower(0,model.initial()[1],0)
        for depth in range(1,9):
            value=lower(depth,*model.initial())
            assert previous<=value<=upper
            row['depths'].append(dict(depth=depth,lower=value))
            previous=value
            if value==upper:
                row.update(certified=True,first_certifying_depth=depth);break
        else:row.update(certified=False,first_certifying_depth=None)
        row['resolved']=True
    except SearchLimit as exc:row.update(resolved=False,certified=False,reason=str(exc))
    row.update(seconds=perf_counter()-start,states=lower.cache_info().misses)
    return row

def run(stage):
    original=json.loads((ROOT/'results/trb_error_20261030/full.json').read_text());assert original['passed']
    policies=json.loads((OUT/'lookahead6_full.json').read_text());assert policies['passed']
    key=lambda r:(r['case_id'],r['capacity'],r['delay'])
    mapping={key(r):r for r in policies['rows']};rows=original['rows'];cache=[]
    if stage=='smoke':
        rows=[next(r for r in rows if r['airport']==a and r['capacity']=='N4' and r['delay']==d)
              for a in ('DEN','PHX') for d in (1,2)]
    else:
        smoke=json.loads((OUT/'certificate_smoke.json').read_text());assert smoke['passed'];cache=smoke['rows']
    done={key(r):r for r in cache}
    design=ROOT/'data/trb_calibrated_20261030'/('certificate_'+stage+'_design.json')
    design.write_text(json.dumps(dict(maximum_depth=8,selection='All frozen report-error conditions',
        upper_policy='Independently evaluated six-step receding policy with matching total request cap',
        stopping='First integer depth at which the admissible lower bound equals the feasible policy value',
        tasks=[key(r) for r in rows]),indent=2),encoding='utf-8')
    tasks=[(r,mapping[key(r)]) for r in rows if key(r) not in done]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,r in enumerate(pool.map(one,tasks),1):
            done[key(r)]=r
            if stage=='smoke' or i%40==0:
                (OUT/('certificate_'+stage+'_progress.json')).write_text(json.dumps(list(done.values()),indent=2),encoding='utf-8')
                print(json.dumps(dict(completed=len(done),total=len(rows))),flush=True)
    result=dict(passed=all(r['resolved'] for r in done.values()),conditions=len(rows),
        certified=sum(r['certified'] for r in done.values()),rows=[done[key(r)] for r in rows])
    (OUT/('certificate_'+stage+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','full']);run(ap.parse_args().stage)
