"""Evaluate every comparator under the identical global report-error set."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
from math import ceil
from pathlib import Path
from time import perf_counter

from robust_reports import RobustReports,prepare,SearchLimit
from rollout_check import RolloutReports
from nominal_failure import stress

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/trb_error_20261030'

def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')

def one(row):
    source=ROOT/row['input_path']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==row['input_sha256']
    c=json.loads(source.read_text(encoding='utf-8'))
    divisor={'N':1,'N4':4,'N8':8}[row['capacity']]
    p=prepare(c['service'],c['n'],(ceil(c['n']/divisor),)*len(c['caps']),c['lag'])
    answer={k:row[k] for k in ('case_id','airport','day','capacity','delay','budget','n')}
    for name in ('independent','rollout2'):
        try:
            if name=='independent':
                model=RobustReports(p,c['lag']+2,row['delay'],row['budget'],1,
                                    independent_errors=True,seconds_limit=12)
            else:
                model=RolloutReports(p,c['lag']+2,row['delay'],row['budget'],1,
                                     depth=2,seconds_limit=12)
            value=model.solve()
            result=stress(model,p)
            assert result['calendars_with_loss']==0
            common=max(x['worst_postrelease'] for x in result['records'])
            assert row['robust']['value']<=common
            if name=='rollout2':
                assert value['value']==common
            answer[name]=dict(common_value=common,minutes=common*c['delta']/c['n'],
                worst_requests=max(x['worst_requests'] for x in result['records']),
                histories=result['histories'],states=value['states'],seconds=value['seconds'],preserved=True)
        except SearchLimit as exc:
            answer[name]=dict(resolved=False,reason=str(exc))
    return answer

def run(stage):
    parent=json.loads((OUT/'full.json').read_text(encoding='utf-8'))
    assert parent['passed']
    rows=parent['rows']
    if stage=='smoke':
        rows=[next(r for r in rows if r['airport']==a and r['capacity']=='N4' and r['delay']==d)
              for a in ('DEN','PHX') for d in (1,2)]
    else:
        assert json.loads((OUT/'matched_smoke.json').read_text())['passed']
    started=perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,result in enumerate(pool.map(one,rows),1):
            results.append(result)
            if stage=='smoke' or i%25==0:
                save(OUT/('matched_'+stage+'_progress.json'),results)
                print(json.dumps(dict(completed=i,total=len(rows),seconds=perf_counter()-started)),flush=True)
    passed=all(all('common_value' in r[k] for k in ('independent','rollout2')) for r in results)
    report=dict(passed=passed,conditions=len(rows),seconds=perf_counter()-started,rows=results)
    save(OUT/('matched_'+stage+'.json'),report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','full'])
    run(ap.parse_args().stage)
