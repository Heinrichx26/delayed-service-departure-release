"""Independently enumerate physical decisions under a hard request ceiling."""
import argparse
import json
from pathlib import Path
import random
from budgeted_reports import BudgetedReports
from robust_reports import prepare
from smoke import exhaustive

OUT=Path(__file__).resolve().parents[2]/'results/trb_error_20261030'


def run(stage):
    OUT.mkdir(parents=True,exist_ok=True)
    if stage=='audit':
        assert json.loads((OUT/'budget_enumeration_smoke.json').read_text())['passed']
    cases=[prepare(((0,0,0,0,2,0,0,0),(0,0,0,0,0,2,0,0)),2,(2,)*8,2)]
    rng=random.Random(20261031)
    while len(cases)<(1 if stage=='smoke' else 12):
        h=rng.randrange(4,7); n=rng.randrange(1,4); lag=rng.randrange(2)
        paths=tuple(tuple(rng.randrange(n+1) if t>=lag else 0 for t in range(h))
                    for _ in range(2))
        try: cases.append(prepare(paths,n,(n,)*h,lag))
        except ValueError: pass
    rows=[]
    for i,p in enumerate(cases):
        for gamma in (0,1):
            for delay in (1,2):
                for cap in (0,1,2):
                    lead=5 if i==0 else min(3,p['h']-1)
                    model=BudgetedReports(p,lead,delay,1,gamma,total_requests=cap)
                    value=model.solve()['value']
                    independent=exhaustive(p,lead,delay,1,gamma,total_requests=cap)
                    assert value==independent,(i,gamma,delay,cap,value,independent)
                    audit=model.verify()
                    assert max(x['worst_requests'] for x in audit['paths'])<=cap
                    rows.append(dict(case=i,gamma=gamma,delay=delay,cap=cap,
                                     exact=value,enumeration=independent))
    result=dict(passed=True,conditions=len(rows),rows=rows)
    (OUT/('budget_enumeration_'+stage+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['smoke','audit'])
    run(parser.parse_args().stage)
