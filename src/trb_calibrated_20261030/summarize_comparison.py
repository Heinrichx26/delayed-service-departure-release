"""Numerical summaries from saved paired experiments; no experiment rerun."""
from pathlib import Path
from statistics import mean
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/trb_calibrated_20261030'
OLD=ROOT/'results/trb_error_20261030'

def read(path):
    result=json.loads(path.read_text(encoding='utf-8'))
    assert result['passed'],path
    return result

def paired(rows,seed):
    assert len(rows)==200
    values=[r['lookahead']['time']-r['exact']['time'] for r in rows]
    assert min(values)>=-1e-9
    clusters=sorted({(r['airport'],r['day']) for r in rows})
    sums=[];counts=[]
    for k in clusters:
        group=[v for r,v in zip(rows,values) if (r['airport'],r['day'])==k]
        sums.append(sum(group));counts.append(len(group))
    rng=np.random.default_rng(seed)
    draws=rng.integers(len(clusters),size=(10000,len(clusters)))
    means=np.asarray(sums)[draws].sum(axis=1)/np.asarray(counts)[draws].sum(axis=1)
    lo,hi=np.quantile(means,[.025,.975])
    a=mean(r['lookahead']['time'] for r in rows)
    b=mean(r['exact']['time'] for r in rows)
    return dict(conditions=len(rows),clusters=len(clusters),lookahead_minutes=a,
        capped_exact_minutes=b,saved_seconds=60*(a-b),saved_percent=100*(a-b)/a,
        saved_seconds_interval=[60*float(lo),60*float(hi)],
        strictly_improved=sum(v>1e-9 for v in values),
        request_ceiling=mean(r['lookahead']['max_requests'] for r in rows),
        exact_max_requests=mean(r['exact']['max_requests'] for r in rows))

def main():
    paths=[OLD/'budget_full.json',OUT/'lookahead4_full.json',OUT/'lookahead6_full.json',
           OUT/'mip_request_ties/mip_audit.json',OUT/'mip_request_ties/mip_public_full.json']
    old=read(paths[0]);datasets={}
    datasets[2]=[dict(case_id=r['case_id'],airport=r['airport'],day=r['day'],
        capacity=r['capacity'],delay=r['delay'],n=r['n'],
        lookahead=dict(time=r['lookahead_time'],max_requests=r['total_request_limit']),
        exact=dict(time=r['time'],max_requests=r['max_requests'])) for r in old['rows']]
    for depth in (4,6):datasets[depth]=read(OUT/f'lookahead{depth}_full.json')['rows']
    key=lambda r:(r['case_id'],r['capacity'],r['delay'])
    assert all({key(r) for r in datasets[d]}=={key(r) for r in datasets[2]} for d in (4,6))
    summary=dict(conditions_per_depth=400,delays_minutes=[2,4],error_allowance=1,
        requests_per_slot=1,matching='Separate maximum whole-period request ceiling for each lookahead policy and condition',
        lookahead={})
    for depth,rows in datasets.items():
        summary['lookahead'][str(depth)]={str(d*2):paired([r for r in rows if r['delay']==d],20261030+depth*10+d) for d in (1,2)}
    mip=read(OUT/'mip_request_ties/mip_public_full.json')['rows'];audit=read(OUT/'mip_request_ties/mip_audit.json')
    summary['mip']=dict(audit_conditions=audit['conditions'],public_conditions=len(mip),
        dp_resolved=len(mip),mip_resolved=sum(r['mip']['resolved'] for r in mip),
        dp_elapsed_range=[min(r['dp']['end_to_end_seconds'] for r in mip),max(r['dp']['end_to_end_seconds'] for r in mip)],
        mip_elapsed_range=[min(r['mip']['end_to_end_seconds'] for r in mip),max(r['mip']['end_to_end_seconds'] for r in mip)],
        worlds_range=[min(r['mip']['worlds'] for r in mip),max(r['mip']['worlds'] for r in mip)],
        variables_range=[min(r['mip']['variables'] for r in mip),max(r['mip']['variables'] for r in mip)])
    summary['input_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (OUT/'comparison_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
