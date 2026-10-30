"""Chronological out-of-sample evaluation of delayed shared-error range release."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from math import ceil
from pathlib import Path
from time import perf_counter
import argparse
import hashlib
import json
import os
import sys

from range_release import requirement, physical, ROOT
DATA=ROOT/'data/trb_information_transfer_20261030'
OUT=ROOT/'results/trb_information_transfer_20261030'


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


@lru_cache(maxsize=150000)
def independent_requirement(upper,n,caps,lag,past,reports):
    changed=list(upper)
    for j,y in reports:
        if y-1>upper[j] or (j<len(past) and abs(past[j]-y)>1):
            return None
        changed[j]=min(changed[j],y+1)
    return requirement(tuple(changed),n,caps,lag,past,(),0)


def replay(actual,upper,n,cap,delay,budget,independent=False,targets=None,policy=None,force_schedule=False):
    h=len(upper);lag=6;caps=(cap,)*h
    early=tuple(min(n,(j+1)*cap) for j in range(h))
    ref=physical(early,actual,lag)
    leaves=[]
    def walk(t,reports,pending,previous,releases,error_used,requests,exhausted):
        if t==h:
            departures=physical(releases,actual,lag)
            loss=tuple(a-b for a,b in zip(ref,departures))
            assert all(x>=0 for x in loss) and releases[-1]==n
            leaves.append(dict(postrelease=sum(a-b for a,b in zip(releases,departures)),
                               gateholding=sum(a-b for a,b in zip(early,releases)),
                               shortfall=sum(loss),maximum_shortfall=max(loss),requests=requests,
                               error_used=error_used,range_exhausted=exhausted))
            return
        b=(independent_requirement(upper,n,caps,lag,actual[:t],reports) if independent
           else requirement(upper,n,caps,lag,actual[:t],reports,1))
        invalid=exhausted or b is None
        z=min(n,previous+cap) if invalid else max(previous,b[t])
        assert 0<=z-previous<=cap
        if policy is not None and not invalid and z<n and requests<budget:
            queried=tuple(sorted(j for j,_ in reports)+sorted(j for _,j in pending))
            queried=tuple(sorted(queried))
            target=policy.select(t,actual[:t],reports,pending,queried,previous)
            request=target is not None
            if request:
                assert t+delay<=target<min(h,t+9) and target not in queried
        else:
            target=(targets[t] if targets is not None and t<len(targets) else t+lag+delay)
            request=bool(policy is None and (force_schedule or (not invalid and z<n)) and t<budget and target<h)
        due=pending+(((t+delay,target),) if request else ())
        arrival=tuple(j for d,j in due if d==t+1)
        pending2=tuple((d,j) for d,j in due if d>t+1)
        assert len(arrival)<=1
        options=[(reports,error_used)]
        if arrival:
            j=arrival[0]
            options=[]
            for error in range(-(1-error_used),2-error_used):
                y=actual[j]+error
                if 0<=y<=n:
                    options.append((tuple(sorted(reports+((j,y),))),error_used+abs(error)))
        for reported,used in options:
            walk(t+1,reported,pending2,z,releases+(z,),used,requests+request,invalid)
    walk(0,(),(),0,(),0,0,False)
    covered=all(a<=u for a,u in zip(actual,upper))
    if covered:
        assert all(r['shortfall']==0 for r in leaves)
    nominal=[r for r in leaves if r['error_used']==0]
    assert len(nominal)==1
    return dict(covered=covered,histories=len(leaves),takeoffs_preserved=all(r['shortfall']==0 for r in leaves),
                worst_shortfall=max(r['shortfall'] for r in leaves),maximum_shortfall=max(r['maximum_shortfall'] for r in leaves),
                worst_postrelease=max(r['postrelease'] for r in leaves),
                minimum_gateholding=min(r['gateholding'] for r in leaves),
                maximum_requests=max(r['requests'] for r in leaves),
                reference_completes_group=ref[-1]==n,nominal=nominal[0])


def design():
    frozen=read(ROOT/'data/trb_transfer_20261030/calibrated_design.json')
    result=dict(frozen,methods=['none','one','three','six','three_independent'],
                request_budgets=[0,1,3,6],answer_delay_slots=[1,2],shared_absolute_error_allowance=1,
                request_rule='One count requested in each initial slot until budget or release completion; target is the first service slot whose lag-adjusted release deadline is not earlier than answer arrival',
                dates_selected_using_test=False,all_test_dates_retained=True,
                paired_outcomes='All common conditions retained, with preservation and worst-history post-release time reported separately')
    save(DATA/'design.json',result)
    return result


def one(task):
    job,days=task
    path=ROOT/job['input_path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==job['input_sha256']
    c=read(path);rows=[]
    upper=tuple(c['calibrated_upper'])
    for v in c['evaluations']:
        if days and v['day'] not in days: continue
        actual=tuple(v['actual'])
        for delay in (1,2):
            methods={name:replay(actual,upper,c['n'],job['cap'],delay,q,independent)
                     for name,q,independent in [('none',0,False),('one',1,False),('three',3,False),
                                                ('six',6,False),('three_independent',3,True)]}
            rows.append(dict(case_id=c['case_id'],airport=c['airport'],hour=c['hour'],n=c['n'],
                             cap=job['cap'],day=v['day'],delay=delay,methods=methods))
    requirement.cache_clear();independent_requirement.cache_clear()
    return rows


def run(stage):
    d=design() if stage=='smoke' else read(DATA/'design.json')
    jobs=d['jobs']
    if stage=='smoke':
        selected=[]
        for airport in ('ATL','CLT','DFW'):
            candidate=max((j for j in jobs if j['airport']==airport),key=lambda j:j['n'])
            selected.extend(j for j in jobs if j['case_id']==candidate['case_id'])
        jobs=selected
        days=['2025-02-07','2025-03-31']
    else:
        smoke=read(OUT/'smoke.json')
        assert smoke['passed'] and smoke['usable_for_full']
        days=None
    assert read(OUT/'range_check.json')['passed']
    os.environ['OMP_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='1'
    rows=[];start=perf_counter()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for block in pool.map(one,[(j,days) for j in jobs]):
            rows.extend(block)
            print(json.dumps(dict(completed=len(rows),seconds=round(perf_counter()-start,2))),flush=True)
            save(OUT/(stage+'_progress.json'),dict(rows=len(rows),seconds=perf_counter()-start))
    common=[r for r in rows if all(r['methods'][m]['reference_completes_group'] and r['methods'][m]['takeoffs_preserved']
                                 for m in ('none','one','three','six','three_independent'))]
    summaries={}
    for name in ('none','one','three','six','three_independent'):
        summaries[name]=dict(preserved=sum(r['methods'][name]['takeoffs_preserved'] for r in rows),
                             worst_loss=max(r['methods'][name]['maximum_shortfall'] for r in rows),
                             mean_holding_seconds=sum(r['methods'][name]['minimum_gateholding']*120/r['n'] for r in common)/len(common),
                             mean_gain_seconds=sum((r['methods']['none']['worst_postrelease']-r['methods'][name]['worst_postrelease'])*120/r['n'] for r in common)/len(common),
                             mean_requests=sum(r['methods'][name]['maximum_requests'] for r in common)/len(common))
    result=dict(passed=True,usable_for_full=bool(common) and summaries['three']['mean_gain_seconds']>0,
                conditions=len(rows),paired_complete_preserved=len(common),seconds=perf_counter()-start,
                methods=summaries,error_histories=sum(v['histories'] for r in rows for v in r['methods'].values()),rows=rows)
    save(OUT/(stage+'.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['smoke','full'])
    run(parser.parse_args().stage)
