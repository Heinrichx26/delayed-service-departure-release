"""Split-calibrated simultaneous service ranges and past-observation release."""
from datetime import date, timedelta
from math import ceil
from pathlib import Path
import argparse
import hashlib
import json
import sys
from zoneinfo import ZoneInfo

import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/trb_error_20261030'))
sys.path.insert(0,str(ROOT/'src/trb_external_error_20261030'))
from robust_reports import physical
from prospective_transfer import profile, AIRPORTS, save, read, sha, DATA, OUT


def minimum_release(service,n,caps,lag):
    early=tuple(min(n,sum(caps[:t+1])) for t in range(len(caps)))
    ref=physical(early,service,lag)
    needed=[ref[min(t+lag,len(caps)-1)] for t in range(len(caps))]
    needed[-1]=n
    for t in range(len(caps)-2,-1,-1):
        needed[t]=max(needed[t],needed[t+1]-caps[t+1])
    return tuple(needed),ref


def release_with_past(actual,upper,n,cap,lag=6):
    h=len(actual); caps=(cap,)*h; releases=[];exhausted=False
    for t in range(h):
        previous=releases[-1] if releases else 0
        if t and actual[t-1]>upper[t-1]:
            exhausted=True
        if exhausted:
            z=min(n,previous+cap)
        else:
            worst=tuple(actual[:t])+tuple(upper[t:])
            needed,_=minimum_release(worst,n,caps,lag)
            z=max(previous,needed[t])
        assert 0<=z-previous<=cap
        releases.append(z)
    assert releases[-1]==n
    early=tuple(min(n,(t+1)*cap) for t in range(h))
    reference=physical(early,actual,lag)
    departures=physical(releases,actual,lag)
    losses=tuple(a-b for a,b in zip(reference,departures))
    covered=all(a<=u for a,u in zip(actual,upper))
    assert all(x>=0 for x in losses)
    if covered:
        assert not any(losses)
    return dict(covered=covered,takeoffs_preserved=not any(losses),
                shortfall_aircraftslots=sum(losses),maximum_shortfall=max(losses),
                final_reference=reference[-1],final_departures=departures[-1],
                reference_completes_group=reference[-1]==n,
                postrelease_aircraftslots=sum(a-b for a,b in zip(releases,departures)),
                early_aircraftslots=sum(a-b for a,b in zip(early,reference)),
                gateholding_aircraftslots=sum(a-b for a,b in zip(early,releases)))


def design():
    paths=[ROOT/'data/independent_adsb_q1/processed/airport_departures_2025_q1.csv',
           ROOT/'open_source_release/data/independent_adsb_q1/processed/airport_departures_2025_q1.csv']
    source=next(p for p in paths if p.is_file())
    frame=pd.read_csv(source,usecols=['airport','departure_utc'])
    frame['utc']=pd.to_datetime(frame.departure_utc,utc=True)
    training=[date(2025,1,d) for d in range(3,19)]
    calibration=[date(2025,1,19)+timedelta(days=k) for k in range(19)]
    evaluation=[date(2025,2,7)+timedelta(days=k) for k in range(53)]
    jobs=[];excluded=[]
    for airport,timezone in AIRPORTS.items():
        f=frame[frame.airport==airport].copy()
        local=f.utc.dt.tz_convert(ZoneInfo(timezone))
        f['day']=local.dt.date;f['minute']=local.dt.hour*60+local.dt.minute+local.dt.second/60
        byday={d:g for d,g in f.groupby('day')};empty=f.iloc[:0]
        for hour in (10,12,14,16,18):
            donor=[profile(byday.get(d,empty),hour) for d in training]
            n=min(sum(c) for c in donor)
            if n<2:
                excluded.append(dict(airport=airport,hour=hour,n=n));continue
            base=[min(n,max(c[j] for c in donor)) for j in range(36)]
            samples=[[min(n,x) for x in profile(byday.get(d,empty),hour)] for d in calibration]
            scores=[max([0]+[x-u for x,u in zip(c,base)]) for c in samples]
            # ceil((m+1)*(1-alpha))=19 for m=19 and alpha=0.05.
            order=ceil((len(samples)+1)*0.95)
            assert order==len(samples)==19
            q=sorted(scores)[order-1]
            upper=[min(n,u+q) if j>=6 else 0 for j,u in enumerate(base)]
            q90=sorted(scores)[ceil((len(samples)+1)*0.90)-1]
            upper90=[min(n,u+q90) if j>=6 else 0 for j,u in enumerate(base)]
            cases=[dict(day=str(d),actual=[min(n,x) for x in profile(byday.get(d,empty),hour)]) for d in evaluation]
            key=f'{airport}_h{hour:02d}'
            case=dict(case_id=key,airport=airport,hour=hour,n=n,lag=6,
                      training_dates=[str(d) for d in training],
                      calibration_dates=[str(d) for d in calibration],
                      calibration_scores=scores,order_statistic=order,radius=q,
                      base_upper=base,calibrated_upper=upper,
                      radius90=q90,calibrated_upper90=upper90,evaluations=cases)
            target=DATA/'calibrated_cases'/(key+'.json');save(target,case)
            for cap in sorted({n,ceil(n/4),ceil(n/8)},reverse=True):
                jobs.append(dict(case_id=key,airport=airport,hour=hour,n=n,cap=cap,
                                 input_path=target.relative_to(ROOT).as_posix(),input_sha256=sha(target)))
    d=dict(source='data/independent_adsb_q1/processed/airport_departures_2025_q1.csv',source_sha256=sha(source),
           training_dates=[str(d) for d in training],calibration_dates=[str(d) for d in calibration],
           evaluation_dates=[str(d) for d in evaluation],
           alpha=0.05,calibration_count=19,order_statistic=19,
           score='Largest positive two-minute count exceedance over training maxima within one hour',
           selection='Group size and airport-hour eligibility use training dates only; all 53 later dates retained',
           coverage_interpretation='Per-airport-hour simultaneous 60-minute count coverage; nominal 95 percent requires exchangeability of calibration and evaluation scores',
           calibration_sensitivity='Raw training maxima and 90 and 95 percent calibrated upper counts evaluated on the same frozen dates',
           missing_records='Zero recorded counts retained; counts describe the public archive',
           decision_period_minutes=72,travel_minutes=12,source_window_minutes=60,slot_minutes=2,
           excluded=excluded,jobs=jobs)
    save(DATA/'calibrated_design.json',d)
    return d


def run(stage):
    d=design() if stage=='smoke' else read(DATA/'calibrated_design.json')
    jobs=d['jobs']
    if stage=='smoke':
        selected=[]
        for airport in AIRPORTS:
            x=max((j for j in jobs if j['airport']==airport),key=lambda j:j['n'])
            selected.extend(j for j in jobs if j['case_id']==x['case_id'])
        jobs=selected
    else:
        assert read(OUT/'calibrated_smoke.json')['passed']
    rows=[]
    for job in jobs:
        c=read(ROOT/job['input_path']);assert sha(ROOT/job['input_path'])==job['input_sha256']
        for v in c['evaluations']:
            if stage=='smoke' and v['day'] not in ('2025-02-07','2025-03-31'):
                continue
            row=dict(job,day=v['day'],radius=c['radius'])
            for name,key in [('raw','base_upper'),('calibrated90','calibrated_upper90'),('calibrated','calibrated_upper')]:
                row[name]=release_with_past(tuple(v['actual']),tuple(c[key]),c['n'],job['cap'])
            rows.append(row)
    result=dict(passed=True,conditions=len(rows),rows=rows)
    save(OUT/('calibrated_'+stage+'.json'),result)
    print(json.dumps(dict(passed=True,conditions=len(rows),
                         raw_preserved=sum(r['raw']['takeoffs_preserved'] for r in rows),
                         calibrated_preserved=sum(r['calibrated']['takeoffs_preserved'] for r in rows),
                         calibrated_covered=sum(r['calibrated']['covered'] for r in rows),
                         maximum_loss=max(r['calibrated']['maximum_shortfall'] for r in rows),
                         calibrated_gateholding_slots=sum(r['calibrated']['gateholding_aircraftslots'] for r in rows))),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','full']);run(ap.parse_args().stage)
