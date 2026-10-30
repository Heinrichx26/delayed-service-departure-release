"""Hold the full request timetable fixed when comparing error accounting."""
from concurrent.futures import ProcessPoolExecutor
from evaluate_ranges import ROOT,OUT,DATA,read,save,replay
import argparse


def one(task):
    job,days=task;c=read(ROOT/job['input_path']);rows=[]
    for v in c['evaluations']:
        if days and v['day'] not in days:continue
        for delay in (1,2):
            values={name:replay(tuple(v['actual']),tuple(c['calibrated_upper']),c['n'],job['cap'],delay,3,
                                independent=independent,force_schedule=True)
                    for name,independent in [('shared',False),('independent',True)]}
            assert all(x['maximum_requests']==3 for x in values.values())
            rows.append(dict(case_id=c['case_id'],cap=job['cap'],n=c['n'],day=v['day'],delay=delay,methods=values))
    return rows


def run(stage):
    d=read(DATA/'design.json');jobs=d['jobs'];days=None
    if stage=='smoke':jobs=[j for j in jobs if j['case_id'] in ('ATL_h14','CLT_h16','DFW_h18')];days=['2025-02-07','2025-03-31']
    else:assert read(OUT/'matched_smoke.json')['passed']
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for block in pool.map(one,[(j,days) for j in jobs]):rows.extend(block)
    key=lambda r:(r['case_id'],r['cap'],r['delay'],r['day'])
    base=read(OUT/'full.json')
    common={key(r) for r in base['rows'] if all(v['reference_completes_group'] and v['takeoffs_preserved'] for v in r['methods'].values())}
    pairs=[r for r in rows if key(r) in common]
    gain=sum((r['methods']['independent']['worst_postrelease']-r['methods']['shared']['worst_postrelease'])*120/r['n'] for r in pairs)/len(pairs)
    out=dict(passed=True,conditions=len(rows),paired_conditions=len(pairs),
             same_request_times_and_targets=True,requests_per_history=3,shared_error_gain_seconds=gain,rows=rows)
    save(OUT/('matched_'+stage+'.json'),out)
    print({k:v for k,v in out.items() if k!='rows'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','full']);run(p.parse_args().stage)
