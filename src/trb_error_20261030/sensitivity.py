"""A fixed first-date two-airport sensitivity panel with Gamma=2."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
from math import ceil
from pathlib import Path
from robust_reports import RobustReports,prepare,SearchLimit
from deadline_cover import solve

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/trb_error_20261030'

def analytic():
    p=prepare(((0,0,0,0,2,0,0,0),(0,0,0,0,0,2,0,0)),2,(2,)*8,2)
    rows=[]
    for g,d in ((0,1),(1,1),(2,1),(1,2)):
        m=RobustReports(p,5,d,1,g)
        value=m.solve();check=m.verify();c=solve(p,5,d,1,g)
        rows.append(dict(gamma=g,delay=d,value=value['value'],requests=c['requests'],target_feasible=c['feasible'],checks=check['physical_checks']))
    assert [r['value'] for r in rows]==[4,4,6,6]
    assert [r['requests'] for r in rows]==[1,2,None,None]
    (OUT/'sensitivity_smoke.json').write_text(json.dumps(dict(passed=True,analytic=rows),indent=2),encoding='utf-8')
    print(json.dumps(rows),flush=True)

def one(r):
    c=json.loads((ROOT/Path(r['input_path'].replace('\\', '/'))).read_text(encoding='utf-8'))
    p=prepare(c['service'],c['n'],(ceil(c['n']/4),)*len(c['caps']),c['lag'])
    answer={k:r[k] for k in ('case_id','airport','day','delay','capacity','n')}
    try:
        m=RobustReports(p,c['lag']+2,r['delay'],1,2,seconds_limit=30,state_limit=350000)
        a=m.solve();v=m.verify()
        assert r['accurate']['value']<=r['robust']['value']<=a['value']
        answer.update(resolved=True,gamma0=r['accurate']['worst_minutes_per_departure'],
                      gamma1=r['robust']['worst_minutes_per_departure'],
                      gamma2=a['value']*c['delta']/c['n'],states=a['states'],
                      seconds=a['seconds'],checks=v['physical_checks'])
    except SearchLimit as e:answer.update(resolved=False,reason=str(e))
    return answer

def public():
    assert json.loads((OUT/'sensitivity_smoke.json').read_text())['passed']
    full=json.loads((OUT/'full.json').read_text())['rows']
    selected=[next(r for r in full if r['airport']==a and r['capacity']=='N4' and r['delay']==d)
              for a in ('DEN','PHX') for d in (1,2)]
    with ProcessPoolExecutor(max_workers=2) as pool:rows=list(pool.map(one,selected))
    report=dict(passed=all(r['resolved'] for r in rows),selection='First date/window in frozen panel, N4 at each airport and both delays',rows=rows)
    (OUT/'sensitivity_public.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','public'])
    a=p.parse_args();analytic() if a.stage=='smoke' else public()
