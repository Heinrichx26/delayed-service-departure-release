"""Chronologically separated shared-error replay with donor-only selection."""
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from math import ceil
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src/trb_external_error_20261030'))
from independent_panel import TimedLookahead
from independent_replay import stress
from budgeted_reports import BudgetedReports
from robust_reports import prepare, physical, SearchLimit
from service_mismatch import bound
DATA = ROOT/'data/trb_transfer_20261030'
OUT = ROOT/'results/trb_transfer_20261030'
AIRPORTS = {'ATL':'America/New_York','CLT':'America/New_York','DFW':'America/Chicago'}


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def profile(frame, hour):
    x = np.floor((frame['minute'].to_numpy()-hour*60)/2).astype(int)
    return [0]*6+np.bincount(x[(x>=0)&(x<30)], minlength=30).tolist()


def design():
    candidates = [ROOT/'data/independent_adsb_q1/processed/airport_departures_2025_q1.csv',
                  ROOT/'open_source_release/data/independent_adsb_q1/processed/airport_departures_2025_q1.csv']
    source = next(p for p in candidates if p.is_file())
    frame = pd.read_csv(source, usecols=['airport','departure_utc'])
    frame['utc'] = pd.to_datetime(frame['departure_utc'], utc=True)
    donors = [date(2025,1,d) for d in range(3,19)]
    evaluations = [date(2025,2,1)+timedelta(days=d) for d in range(59)]
    jobs, excluded, windows = [], [], []
    for airport, timezone in AIRPORTS.items():
        f = frame[frame.airport==airport].copy()
        local = f['utc'].dt.tz_convert(ZoneInfo(timezone))
        f['day'] = local.dt.date
        f['minute'] = local.dt.hour*60+local.dt.minute+local.dt.second/60
        byday = {day:g for day,g in f.groupby('day')}
        empty = f.iloc[:0]
        for hour in (10,12,14,16,18):
            calendars = [profile(byday.get(day,empty),hour) for day in donors]
            n = min(sum(c) for c in calendars)
            if n<2:
                excluded.append(dict(airport=airport,hour=hour,n=n,reason='Donor minimum group below two'))
                continue
            key = f'{airport}_h{hour:02d}'
            case = dict(key=key,airport=airport,hour=hour,n=n,lag=6,delta=2,
                        donor_dates=[str(d) for d in donors],
                        service=[[min(n,x) for x in c] for c in calendars],
                        evaluations=[dict(day=str(day),service=[min(n,x) for x in profile(byday.get(day,empty),hour)])
                                     for day in evaluations])
            target = DATA/'cases'/(key+'.json')
            save(target,case)
            windows.extend(dict(airport=airport,hour=hour,day=str(d)) for d in evaluations)
            for cap in sorted({n,ceil(n/4),ceil(n/8)},reverse=True):
                try:
                    prepare(case['service'],n,(cap,)*36,6)
                except ValueError:
                    excluded.append(dict(airport=airport,hour=hour,n=n,cap=cap,reason='Donor-incomplete release setting'))
                    continue
                for delay in (1,2):
                    jobs.append(dict(case_id=key,airport=airport,hour=hour,n=n,cap=cap,delay=delay,
                                     input_path=target.relative_to(ROOT).as_posix(),input_sha256=sha(target)))
    report = dict(source='data/independent_adsb_q1/processed/airport_departures_2025_q1.csv',
                  source_sha256=sha(source),donor_dates=[str(d) for d in donors],
                  evaluation_dates=[str(d) for d in evaluations],
                  selection='Airport hours and release settings determined exclusively from January donors; every February and March date retained',
                  evaluation_used_for_selection=False,excluded=excluded,windows=len(windows),jobs=jobs,
                  error_allowance=1,requests_per_slot=1,slot_minutes=2,
                  report_model='Actual future count with all integer residual histories of total absolute error at most one',
                  recorded_service='Observed departures in the public archive; absent records retained as zero recorded counts',
                  fallback='After support exhaustion, release at the physical slot limit and issue no further requests',
                  primary_comparison='Six-step and exact at matched donor-support maximum requests; all outcomes kept',
                  decision_period_minutes=72,source_count_window_minutes=60,travel_minutes=12)
    save(DATA/'design.json',report)
    print(json.dumps(dict(jobs=len(jobs),windows=len(windows),excluded=excluded)),flush=True)
    return report


def evaluate(model, p, actual):
    leaves = []
    h,n,lag = p['h'],p['n'],p['lag']
    def walk(t,e,pending,queried,u,used,releases,first_fallback,request_count):
        if t==h:
            v = bound(releases,actual,p)
            done,ref = v['departures'],v['reference']
            leaves.append(dict(shortfall=sum(v['loss']),max_shortfall=max(v['loss']),
                               upper_sum=sum(v['upper']),upper_max=max(v['upper']),
                               postrelease=sum(a-b for a,b in zip(releases,done)),
                               requests=request_count,first_fallback=first_fallback,
                               final_reference=ref[-1],final_departures=done[-1],
                               error_used=used))
            return
        exhausted = not any(x>=0 for x in e)
        if exhausted:
            batch=()
            z=min(n,u+p['caps'][t])
            if first_fallback is None:
                first_fallback=t
        else:
            model.value(t,e,pending,queried,u)
            batch=model.choice[(t,e,pending,queried,u)]
            z=model.release(t,e,u)
        assert 0<=z-u<=p['caps'][t] and len(batch)<=1
        total=request_count+len(batch)
        if hasattr(model,'total_requests'):
            assert total<=model.total_requests
        combined=pending+tuple((t+model.delay,j) for j in batch)
        due=tuple(j for d,j in combined if d==t+1)
        future=tuple((d,j) for d,j in combined if d>t+1)
        assert len(due)<=1
        q=queried|sum(1<<j for j in batch)
        q&=~((1<<min(h,t+1+model.delay))-1)
        q=q&((1<<h)-1)
        if hasattr(model,'total_requests'):
            q|=total<<h
        answers=[((),used)] if not due else [(tuple(actual[j] for j in due),used)]
        if due and used==0:
            answers += [((actual[due[0]]+r,),1) for r in (-1,1) if 0<=actual[due[0]]+r<=n]
        for answer,spent in answers:
            updated=[]
            for s,old in enumerate(e):
                residual=sum(abs(p['paths'][s][j]-y) for j,y in zip(due,answer))
                new=old+residual
                updated.append(new if old>=0 and p['paths'][s][t]==actual[t] and new<=1 else -1)
            walk(t+1,tuple(updated),future,q,z,spent,releases+(z,),first_fallback,total)
    walk(0,(0,)*len(p['paths']),(),0,0,0,(),None,0)
    nominal=[r for r in leaves if r['error_used']==0]
    assert len(nominal)==1
    return dict(histories=len(leaves),all_takeoffs_preserved=all(r['shortfall']==0 for r in leaves),
                worst_shortfall=max(r['shortfall'] for r in leaves),
                worst_max_shortfall=max(r['max_shortfall'] for r in leaves),
                worst_postrelease=max(r['postrelease'] for r in leaves),
                worst_upper_sum=max(r['upper_sum'] for r in leaves),
                worst_upper_max=max(r['upper_max'] for r in leaves),
                support_exhausted_histories=sum(r['first_fallback'] is not None for r in leaves),
                minimum_fallback_slot=min((r['first_fallback'] for r in leaves if r['first_fallback'] is not None),default=None),
                nominal=nominal[0],reference_completes_group=nominal[0]['final_reference']==n,
                actual_in_support=tuple(actual) in p['paths'],bound_verified=True)


def one(task):
    job,days=task
    c=read(ROOT/job['input_path'])
    assert sha(ROOT/job['input_path'])==job['input_sha256']
    p=prepare(c['service'],c['n'],(job['cap'],)*36,6)
    result=dict(job,resolved=False,rows=[])
    try:
        six=TimedLookahead(p,8,job['delay'],1,1,depth=6,seconds_limit=15,state_limit=200000)
        v=six.solve()
        check=stress(six,p)
        q=max(r['worst_requests'] for r in check['records'])
        exact=BudgetedReports(p,8,job['delay'],1,1,total_requests=q,seconds_limit=15,state_limit=200000)
        e=exact.solve()
        assert e['value']<=v['value']
        donor_check=stress(exact,p)
        assert donor_check['calendars_with_loss']==check['calendars_with_loss']==0
        result.update(total_request_limit=q,six_donor_value=v['value'],exact_donor_value=e['value'])
        # Replay only histories compatible with the frozen policy model or its stated fallback.
        for row in c['evaluations']:
            if days and row['day'] not in days:
                continue
            six.started=exact.started=perf_counter()
            sr=evaluate(six,p,tuple(row['service']))
            er=evaluate(exact,p,tuple(row['service']))
            result['rows'].append(dict(day=row['day'],six=sr,exact=er))
        result['resolved']=True
    except SearchLimit as exc:
        result['reason']=str(exc)
    return result


def run(stage):
    d=design() if stage=='smoke' else read(DATA/'design.json')
    if stage=='smoke':
        selected=[]
        for airport in AIRPORTS:
            candidate=max((j for j in d['jobs'] if j['airport']==airport),key=lambda j:j['n'])
            selected.extend(j for j in d['jobs'] if j['case_id']==candidate['case_id'] and j['cap']==ceil(j['n']/4))
        tasks=[(j,['2025-02-01','2025-03-31']) for j in selected]
    else:
        assert read(OUT/'smoke.json')['passed']
        tasks=[(j,None) for j in d['jobs']]
    assert read(OUT/'bound_smoke.json')['passed']
    os.environ['OMP_NUM_THREADS']=os.environ['OPENBLAS_NUM_THREADS']='1'
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for r in pool.map(one,tasks):
            rows.append(r)
            if stage=='smoke' or len(rows)%10==0:
                save(OUT/(stage+'_progress.json'),rows)
                print(json.dumps(dict(completed=len(rows),total=len(tasks),resolved=sum(r['resolved'] for r in rows))),flush=True)
    result=dict(passed=all(r['resolved'] for r in rows),conditions=len(rows),rows=rows)
    save(OUT/(stage+'.json'),result)
    flat=[v for r in rows for v in r['rows']]
    print(json.dumps(dict(passed=result['passed'],conditions=len(rows),date_conditions=len(flat),
                         exact_preserved=sum(v['exact']['all_takeoffs_preserved'] for v in flat),
                         six_preserved=sum(v['six']['all_takeoffs_preserved'] for v in flat),
                         bound_checks=sum(v[m]['histories'] for v in flat for m in ('exact','six')))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['smoke','full'])
    run(parser.parse_args().stage)
