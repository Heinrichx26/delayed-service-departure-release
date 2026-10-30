"""Frozen-training information selection with range-safe out-of-date evaluation."""
from concurrent.futures import ProcessPoolExecutor
from math import ceil
from time import perf_counter
import argparse
import os
from evaluate_ranges import read,save,ROOT,DATA,OUT,replay
from range_information import RangeInformation,SearchLimit
from range_release import requirement


def one(task):
    job,days=task
    c=read(ROOT/job['input_path'])
    donor=read(ROOT/'data/trb_transfer_20261030/cases'/(c['case_id']+'.json'))
    rows=[];methods={};models={};start=perf_counter()
    for name,depth in [('two',2),('six',6),('exact',None)]:
        m=RangeInformation(donor['service'],c['calibrated_upper'],c['n'],job['cap'],job['delay'],
                           budget=3,depth=depth,seconds_limit=20)
        try:
            record=m.solve()
            verified=[replay(tuple(p),tuple(c['calibrated_upper']),c['n'],job['cap'],job['delay'],3,policy=m)
                      for p in donor['service']]
            assert all(r['takeoffs_preserved'] for r in verified)
            assert max(r['worst_postrelease'] for r in verified)==record['value']
            record.update(resolved=True,donor_histories=sum(r['histories'] for r in verified))
            models[name]=m
        except SearchLimit as exc:
            record=dict(resolved=False,reason=str(exc))
        methods[name]=record
    if all(r['resolved'] for r in methods.values()):
        assert methods['exact']['value']<=methods['two']['value'] and methods['exact']['value']<=methods['six']['value']
        for v in c['evaluations']:
            if days and v['day'] not in days:continue
            actual=tuple(v['actual']);upper=tuple(c['calibrated_upper'])
            results={name:replay(actual,upper,c['n'],job['cap'],job['delay'],3,policy=m)
                     for name,m in models.items()}
            results['deadline']=replay(actual,upper,c['n'],job['cap'],job['delay'],3)
            rows.append(dict(case_id=c['case_id'],airport=c['airport'],hour=c['hour'],n=c['n'],cap=job['cap'],
                             delay=job['delay'],day=v['day'],methods=results))
    requirement.cache_clear()
    return dict(job=job,resolved=all(r['resolved'] for r in methods.values()),training=methods,rows=rows,seconds=perf_counter()-start)


def run(stage):
    d=read(DATA/'design.json')
    jobs=[dict(j,delay=delay) for j in d['jobs'] for delay in (1,2)]
    if stage=='smoke':
        chosen=[]
        for airport in ('ATL','CLT','DFW'):
            candidate=max((j for j in jobs if j['airport']==airport),key=lambda j:j['n'])
            chosen.extend(j for j in jobs if j['case_id']==candidate['case_id'] and j['cap']==ceil(j['n']/4))
        jobs=chosen;days=['2025-02-07','2025-03-31']
    else:
        assert read(OUT/'adaptive_smoke.json')['passed']
        days=None
    os.environ['OMP_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='1'
    blocks=[];start=perf_counter()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for b in pool.map(one,[(j,days) for j in jobs]):
            blocks.append(b)
            print(dict(completed=len(blocks),total=len(jobs),resolved=sum(x['resolved'] for x in blocks),seconds=perf_counter()-start),flush=True)
            save(OUT/('adaptive_'+stage+'_progress.json'),dict(completed=len(blocks),total=len(jobs)))
    rows=[r for b in blocks for r in b['rows']]
    common=[r for r in rows if all(v['reference_completes_group'] and v['takeoffs_preserved'] for v in r['methods'].values())]
    summary={name:dict(preserved=sum(r['methods'][name]['takeoffs_preserved'] for r in rows),
                       mean_holding_seconds=sum(r['methods'][name]['minimum_gateholding']*120/r['n'] for r in common)/len(common),
                       mean_postrelease_seconds=sum(r['methods'][name]['worst_postrelease']*120/r['n'] for r in common)/len(common),
                       mean_requests=sum(r['methods'][name]['maximum_requests'] for r in common)/len(common))
             for name in ('two','six','exact','deadline')} if common else {}
    save(OUT/('adaptive_'+stage+'.json'),dict(passed=all(b['resolved'] for b in blocks),blocks=blocks,conditions=len(rows),
         paired_complete_preserved=len(common),summary=summary,seconds=perf_counter()-start))
    print(dict(conditions=len(rows),summary=summary,paired=len(common)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','full']);run(p.parse_args().stage)
