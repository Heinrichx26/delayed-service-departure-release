"""Frozen paired evaluation under identical per-slot and total request limits."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
from math import ceil
from pathlib import Path
from time import perf_counter
from robust_reports import prepare,SearchLimit
from budgeted_reports import BudgetedReports

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/trb_error_20261030'

def one(pair):
    r,m=pair
    c=json.loads((ROOT/Path(r['input_path'].replace('\\', '/'))).read_text(encoding='utf-8'))
    divisor={'N':1,'N4':4,'N8':8}[r['capacity']]
    p=prepare(c['service'],c['n'],(ceil(c['n']/divisor),)*len(c['caps']),c['lag'])
    cap=m['rollout2']['worst_requests']
    answer={k:r[k] for k in ('case_id','airport','day','capacity','delay','n')}
    answer.update(total_request_limit=cap,lookahead_time=m['rollout2']['minutes'],
                  unrestricted_time=r['robust']['worst_minutes_per_departure'])
    try:
        model=BudgetedReports(p,c['lag']+2,r['delay'],1,1,total_requests=cap,
                              seconds_limit=20,state_limit=250000)
        v=model.solve();audit=model.verify()
        maximum=max(x['worst_requests'] for x in audit['paths'])
        assert maximum<=cap
        assert r['robust']['value']<=v['value']<=m['rollout2']['common_value']
        answer.update(resolved=True,value=v['value'],time=v['value']*c['delta']/c['n'],
                      max_requests=maximum,states=v['states'],checks=audit['physical_checks'],seconds=v['seconds'])
    except SearchLimit as e:answer.update(resolved=False,reason=str(e))
    return answer

def run(stage):
    first=json.loads((OUT/'full.json').read_text())
    second=json.loads((OUT/'matched_full.json').read_text())
    assert first['passed'] and second['passed']
    key=lambda r:(r['case_id'],r['capacity'],r['delay'])
    matched={key(r):r for r in second['rows']}
    original=first['rows']
    if stage=='smoke':
        original=[next(r for r in original if r['airport']==a and r['capacity']=='N4' and r['delay']==d)
                  for a in ('DEN','PHX') for d in (1,2)]
    else:assert json.loads((OUT/'budget_smoke.json').read_text())['passed']
    start=perf_counter();rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,row in enumerate(pool.map(one,[(r,matched[key(r)]) for r in original]),1):
            rows.append(row)
            if stage=='smoke' or i%40==0:
                (OUT/('budget_'+stage+'_progress.json')).write_text(json.dumps(rows,indent=2),encoding='utf-8')
                print(json.dumps(dict(completed=i,total=len(original),seconds=perf_counter()-start)),flush=True)
    result=dict(passed=all(r['resolved'] for r in rows),conditions=len(rows),seconds=perf_counter()-start,rows=rows)
    (OUT/('budget_'+stage+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','full'])
    run(ap.parse_args().stage)
