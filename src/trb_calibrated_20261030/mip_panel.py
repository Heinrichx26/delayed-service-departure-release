"""Small direct-MIP smoke check before any larger algorithm comparison."""
from pathlib import Path
import sys,json,argparse,random
from math import ceil
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/trb_error_20261030'))
from robust_reports import prepare,RobustReports
from budgeted_reports import BudgetedReports
from scenario_mip import ScenarioMIP

def smoke():
    p=prepare(((0,0,0,0,2,0,0,0),(0,0,0,0,0,2,0,0)),2,(2,)*8,2)
    rows=[]
    for gamma,cap in [(0,None),(1,None),(1,1),(1,2)]:
        exact=(RobustReports(p,5,1,1,gamma) if cap is None else
               BudgetedReports(p,5,1,1,gamma,total_requests=cap))
        reference=exact.solve();mip=ScenarioMIP(p,5,1,1,gamma,total_requests=cap,seconds_limit=30).solve()
        assert mip['resolved'] and mip['value']==reference['value'],(gamma,cap,mip,reference)
        rows.append(dict(gamma=gamma,cap=cap,exact=reference,mip=mip))
    out=ROOT/'results/trb_calibrated_20261030/mip_request_ties';out.mkdir(parents=True,exist_ok=True)
    (out/'mip_smoke.json').write_text(json.dumps(dict(passed=True,rows=rows),indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,rows=rows)),flush=True)

def audit():
    out=ROOT/'results/trb_calibrated_20261030/mip_request_ties'
    assert json.loads((out/'mip_smoke.json').read_text())['passed']
    rng=random.Random(20261030);cases=[]
    while len(cases)<24:
        h=rng.randrange(4,7);n=rng.randrange(1,4);lag=rng.randrange(2)
        paths=tuple(tuple(rng.randrange(n+1) if t>=lag else 0 for t in range(h)) for _ in range(2))
        try:cases.append(prepare(paths,n,(n,)*h,lag))
        except ValueError:pass
    rows=[]
    for i,p in enumerate(cases):
        for gamma in (0,1):
            for delay in (1,2):
                cap=None if i%2==0 else 1
                exact=(RobustReports(p,3,delay,1,gamma) if cap is None else
                       BudgetedReports(p,3,delay,1,gamma,total_requests=cap))
                reference=exact.solve();m=ScenarioMIP(p,3,delay,1,gamma,total_requests=cap,seconds_limit=20).solve()
                assert m['resolved'] and m['value']==reference['value'],(i,gamma,delay,m,reference)
                rows.append(dict(case=i,gamma=gamma,delay=delay,total_requests=cap,dp=reference,mip=m))
    result=dict(passed=True,conditions=len(rows),rows=rows)
    (out/'mip_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


def public_one(r):
    c=json.loads((ROOT/Path(r['input_path'].replace('\\','/'))).read_text())
    p=prepare(c['service'],c['n'],(ceil(c['n']/4),)*len(c['caps']),c['lag'])
    start=perf_counter();model=RobustReports(p,c['lag']+2,r['delay'],1,1,seconds_limit=20,state_limit=500000)
    dp=model.solve();dp['end_to_end_seconds']=perf_counter()-start
    assert model.verify()['passed']
    start=perf_counter();m=ScenarioMIP(p,c['lag']+2,r['delay'],1,1,seconds_limit=20).solve()
    m['end_to_end_seconds']=perf_counter()-start
    if m['resolved']:assert m['value']==dp['value']
    assert m['dual_bound'] is None or m['dual_bound']<=dp['value']+1e-6
    return dict(case_id=r['case_id'],airport=r['airport'],day=r['day'],delay=r['delay'],n=c['n'],dp=dp,mip=m)


def public(stage):
    out=ROOT/'results/trb_calibrated_20261030/mip_request_ties'
    assert json.loads((out/'mip_audit.json').read_text())['passed']
    original=json.loads((ROOT/'results/trb_error_20261030/full.json').read_text())['rows']
    selected=[]
    for airport in ('DEN','PHX'):
        for year in ('2024','2025'):
            r=next(r for r in original if r['airport']==airport and r['day'].startswith(year) and r['capacity']=='N4' and r['delay']==1)
            for delay in (1,2):selected.append(dict(r,delay=delay))
    if stage=='public_smoke':selected=selected[:2]
    else:assert json.loads((out/'mip_public_smoke.json').read_text())['passed']
    data=ROOT/'data/trb_calibrated_20261030/mip_request_ties';data.mkdir(parents=True,exist_ok=True)
    (data/('mip_'+stage+'_design.json')).write_text(json.dumps(dict(selection='First N/4 window per airport and year, both response delays',seconds_per_method=20,jobs=selected),indent=2),encoding='utf-8')
    cached=[] if stage=='public_smoke' else json.loads((out/'mip_public_smoke.json').read_text())['rows']
    by_key={(r['case_id'],r['delay']):r for r in cached}
    pending=[r for r in selected if (r['case_id'],r['delay']) not in by_key]
    with ProcessPoolExecutor(max_workers=2) as pool: new=list(pool.map(public_one,pending))
    by_key.update({(r['case_id'],r['delay']):r for r in new})
    rows=[by_key[(r['case_id'],r['delay'])] for r in selected]
    result=dict(passed=True,conditions=len(rows),rows=rows)
    (out/('mip_'+stage+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,conditions=len(rows),dp_solved=len(rows),mip_solved=sum(r['mip']['resolved'] for r in rows))),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','audit','public_smoke','public_full'])
    stage=ap.parse_args().stage
    if stage=='smoke':smoke()
    elif stage=='audit':audit()
    else:public(stage)
