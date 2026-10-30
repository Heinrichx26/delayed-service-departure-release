"""Matched report-error experiment with a longer receding lookahead."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
from math import ceil
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/trb_error_20261030'))
from robust_reports import prepare,SearchLimit
from rollout_check import RolloutReports
from budgeted_reports import BudgetedReports
from nominal_failure import stress

OUT=ROOT/'results/trb_calibrated_20261030'
DATA=ROOT/'data/trb_calibrated_20261030'

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

def one(task):
    row,depth=task
    src=ROOT/Path(row['input_path'].replace('\\','/'))
    assert hashlib.sha256(src.read_bytes()).hexdigest()==row['input_sha256']
    c=json.loads(src.read_text(encoding='utf-8'))
    divisor={'N':1,'N4':4,'N8':8}[row['capacity']]
    p=prepare(c['service'],c['n'],(ceil(c['n']/divisor),)*len(c['caps']),c['lag'])
    answer={k:row[k] for k in ('case_id','airport','day','capacity','delay','n')}
    answer['depth']=depth
    try:
        start=perf_counter()
        rollout=RolloutReports(p,c['lag']+2,row['delay'],1,1,depth=depth,
                               seconds_limit=30,state_limit=500000)
        rv=rollout.solve();ra=stress(rollout,p)
        assert ra['calendars_with_loss']==0
        worst=max(x['worst_postrelease'] for x in ra['records'])
        assert worst==rv['value']
        cap=max(x['worst_requests'] for x in ra['records'])
        answer['lookahead']=dict(value=worst,time=worst*c['delta']/c['n'],
            max_requests=cap,states=rv['states'],seconds=rv['seconds'],
            validated_seconds=perf_counter()-start,histories=ra['histories'])
        start=perf_counter()
        exact=BudgetedReports(p,c['lag']+2,row['delay'],1,1,total_requests=cap,
                              seconds_limit=30,state_limit=500000)
        ev=exact.solve();ea=exact.verify()
        used=max(x['worst_requests'] for x in ea['paths'])
        assert used<=cap
        assert row['robust']['value']<=ev['value']<=worst
        answer['exact']=dict(value=ev['value'],time=ev['value']*c['delta']/c['n'],
            max_requests=used,states=ev['states'],seconds=ev['seconds'],
            validated_seconds=perf_counter()-start,checks=ea['physical_checks'])
        answer['resolved']=True
    except SearchLimit as exc:
        answer.update(resolved=False,reason=str(exc))
    return answer

def run(stage,depth):
    source=json.loads((ROOT/'results/trb_error_20261030/full.json').read_text())
    assert source['passed']
    rows=source['rows'];cache=[]
    label=f'lookahead{depth}_{stage}'
    if stage=='smoke':
        rows=[next(r for r in rows if r['airport']==a and r['capacity']=='N4' and r['delay']==d)
              for a in ('DEN','PHX') for d in (1,2)]
    else:
        smoke=json.loads((OUT/f'lookahead{depth}_smoke.json').read_text())
        assert smoke['passed'];cache=smoke['rows']
    key=lambda r:(r['case_id'],r['capacity'],r['delay'])
    done={key(r):r for r in cache}
    tasks=[(r,depth) for r in rows if key(r) not in done]
    save(DATA/(label+'_design.json'),dict(depth=depth,conditions=len(rows),
        selection='Every condition from the frozen 400-condition report-error panel; smoke uses the first N/4 window per airport and both delays',
        error_allowance=1,requests_per_slot=1,seconds_per_method=30,
        matching='Exact total-request ceiling equals the lookahead policy maximum over every allowed history',
        keys=[key(r) for r in rows]))
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    start=perf_counter()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,r in enumerate(pool.map(one,tasks),1):
            done[key(r)]=r
            if stage=='smoke' or i%25==0:
                save(OUT/(label+'_progress.json'),list(done.values()))
                print(json.dumps(dict(completed=len(done),total=len(rows),seconds=perf_counter()-start)),flush=True)
    result=dict(passed=all(done[key(r)]['resolved'] for r in rows),
                conditions=len(rows),seconds=perf_counter()-start,rows=[done[key(r)] for r in rows])
    save(OUT/(label+'.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','full'])
    ap.add_argument('--depth',type=int,default=4);args=ap.parse_args()
    run(args.stage,args.depth)
