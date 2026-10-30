"""Independent-source report-error validation with donor-only group sizes."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from math import ceil
from pathlib import Path
from time import perf_counter
import argparse, hashlib, json, os, sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/trb_error_20261030'))
from robust_reports import RobustReports, SearchLimit, prepare
from budgeted_reports import BudgetedReports
from rollout_check import RolloutReports
from independent_replay import stress
DATA=ROOT/'data/trb_external_error_20261030'
OUT=ROOT/'results/trb_external_error_20261030'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


class TimedLookahead(RolloutReports):
    def __init__(self,*a,**kw):
        self.look_calls=0
        super().__init__(*a,**kw)
    def look(self,*a):
        self.look_calls+=1
        if self.look_calls>300000 or perf_counter()-self.started>self.seconds_limit:
            raise SearchLimit('lookahead resource allowance reached')
        return super().look(*a)
    def value(self,*a):
        if perf_counter()-self.started>self.seconds_limit:
            raise SearchLimit('lookahead time allowance reached')
        return super().value(*a)


def design():
    jobs=[];excluded=[];source=[]
    for airport in ('ATL','CLT','DFW'):
        for path in sorted((ROOT/'data/independent_adsb/cases'/airport).glob('*.json')):
            c=read(path)
            assert c['airport']==airport and len(c['service'])==16
            assert all(day<c['day'] for day in c['donor_days'])
            n=min(sum(p) for p in c['service'])
            source.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),original_n=c['n'],donor_only_n=n))
            if n<2:continue
            case=dict(case_id=c['case_id'],airport=airport,day=c['day'],hour=c['hour'],n=n,
                      lag=c['lag'],delta=c['delta'],donor_days=c['donor_days'],
                      service=[[min(n,x) for x in p] for p in c['service']],
                      source_case=path.relative_to(ROOT).as_posix(),source_case_sha256=sha(path))
            target=DATA/'cases'/airport/path.name
            save(target,case)
            caps={}
            for label,div in [('N',1),('N4',4),('N8',8)]:
                k=ceil(n/div)
                caps.setdefault(k,[]).append(label)
            for cap,labels in caps.items():
                try:prepare(case['service'],n,(cap,)*len(case['service'][0]),case['lag'])
                except ValueError:
                    excluded.append(dict(case_id=c['case_id'],airport=airport,cap=cap,reason='Donor calendars cannot all complete the group'));continue
                for delay in (1,2):
                    jobs.append(dict(case_id=c['case_id'],airport=airport,day=c['day'],hour=c['hour'],n=n,
                                     cap=cap,capacity_labels=labels,delay=delay,
                                     input_path=target.relative_to(ROOT).as_posix(),input_sha256=sha(target)))
    jobs.sort(key=lambda r:(r['airport'],r['day'],r['hour'],-r['cap'],r['delay']))
    selected=[]
    for airport in ('ATL','CLT','DFW'):
        candidates=[r for r in jobs if r['airport']==airport and 'N4' in r['capacity_labels']]
        for subset in [sorted(candidates,key=lambda r:(r['day'],r['hour'],r['delay'])),
                       sorted(candidates,key=lambda r:(-r['n'],r['day'],r['hour'],r['delay']))]:
            first=subset[0]
            selected += [r for r in candidates if r['case_id']==first['case_id']]
    key=lambda r:(r['case_id'],r['cap'],r['delay'])
    smoke=list({key(r):r for r in selected}.values())
    report=dict(selection='All frozen independent January airport-hour windows; group size from 16 donor dates only; unique physical release limits N, ceil(N/4), ceil(N/8); exclude donor-incomplete combinations before solving',
                smoke_selection='First chronological and largest-group N/4 window at each airport, both response times',
                group_size_uses_evaluation_service=False,error_allowance=1,request_limit=1,lookahead_depths=[2,4,6],
                seconds_per_method=15,state_allowance=200000,lookahead_call_allowance=300000,
                physical_service_clipping='Count capped at group size; leaves cumulative takeoffs unchanged for releases <= N',
                all_conditions_retained_after_solving=True,sources=source,excluded=excluded,jobs=jobs,smoke=smoke)
    unique={}
    for row in jobs:
        c=read(ROOT/row['input_path'])
        signature=hashlib.sha256(json.dumps([c['service'],c['n'],c['lag'],row['cap'],row['delay']],sort_keys=True).encode()).hexdigest()
        if signature not in unique:
            unique[signature]=dict(row,numerical_signature=signature,represented_windows=[])
        unique[signature]['represented_windows'].append(dict(case_id=row['case_id'],airport=row['airport'],day=row['day'],hour=row['hour']))
    for airport in ('ATL','CLT','DFW'):
        first=min((r for r in unique.values() if r['airport']==airport),key=lambda r:(r['day'],r['hour']))
        smoke += [r for r in unique.values() if r['case_id']==first['case_id']]
    smoke=list({key(r):r for r in smoke}.values())
    report['smoke']=smoke
    report['smoke_selection']='First chronological window at every distinct release limit plus first and largest-group N/4 windows at each airport, both response times'
    report.update(represented_window_conditions=len(jobs),numerical_deduplication='Identical service calendars, group size, travel, release limit and response time count once',jobs=list(unique.values()))
    save(DATA/'design.json',report)
    print(json.dumps(dict(source_windows=len(source),conditions=len(report['jobs']),represented_window_conditions=len(jobs),smoke_conditions=len(report['smoke']),
                          donor_only_sizes_changed=sum(r['original_n']!=r['donor_only_n'] for r in source),
                          n_range=[min(r['n'] for r in jobs),max(r['n'] for r in jobs)])),flush=True)
    return report


def one(job):
    path=ROOT/job['input_path'];assert sha(path)==job['input_sha256'];c=read(path)
    p=prepare(c['service'],c['n'],(job['cap'],)*len(c['service'][0]),c['lag'])
    row=dict(job);row['methods']={};start=perf_counter();cached={}
    lead=c['lag']+2
    for depth in (2,4,6):
        record={};row['methods'][str(depth)]=record
        try:
            t=perf_counter();m=TimedLookahead(p,lead,job['delay'],1,1,depth=depth,
                                           seconds_limit=15,state_limit=200000)
            value=m.solve();check=stress(m,p)
            assert check['calendars_with_loss']==0
            worst=max(r['worst_postrelease'] for r in check['records'])
            assert worst==value['value']
            cap=max(r['worst_requests'] for r in check['records'])
            record['lookahead']=dict(value=worst,minutes=worst*c['delta']/c['n'],max_requests=cap,
                                     seconds=perf_counter()-t,states=value['states'],histories=check['histories'])
            if cap not in cached:
                t=perf_counter();exact=BudgetedReports(p,lead,job['delay'],1,1,total_requests=cap,
                                                     seconds_limit=15,state_limit=200000)
                v=exact.solve();a=stress(exact,p);assert a['calendars_with_loss']==0
                assert max(r['worst_postrelease'] for r in a['records'])==v['value']
                maximum=max(r['worst_requests'] for r in a['records']);assert maximum<=cap
                cached[cap]=dict(value=v['value'],minutes=v['value']*c['delta']/c['n'],max_requests=maximum,
                                 seconds=perf_counter()-t,states=v['states'],histories=a['histories'])
            record['exact']=cached[cap]
            assert record['exact']['value']<=worst
            record['resolved']=True
        except SearchLimit as exc:record.update(resolved=False,reason=str(exc))
    row.update(resolved=all(r['resolved'] for r in row['methods'].values()),seconds=perf_counter()-start)
    return row


def run(stage):
    d=design() if stage=='smoke' else read(DATA/'design.json')
    jobs=d['smoke'] if stage=='smoke' else d['jobs'];cache=[]
    if stage=='full':
        s=read(OUT/'smoke.json');assert s['passed'] and all(r['resolved'] for r in s['rows']);cache=s['rows']
    key=lambda r:(r['case_id'],r['cap'],r['delay'])
    done={key(r):r for r in cache};tasks=[r for r in jobs if key(r) not in done]
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    start=perf_counter()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,r in enumerate(pool.map(one,tasks),1):
            done[key(r)]=r
            if stage=='smoke' or i%10==0:
                save(OUT/(stage+'_progress.json'),list(done.values()))
                print(json.dumps(dict(completed=len(done),total=len(jobs),resolved=sum(x['resolved'] for x in done.values()),seconds=perf_counter()-start)),flush=True)
    rows=[dict(r, **done[key(r)]) for r in jobs]
    result=dict(passed=all(r['resolved'] for r in rows),conditions=len(rows),resolved=sum(r['resolved'] for r in rows),seconds=perf_counter()-start,rows=rows)
    save(OUT/(stage+'.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','full']);run(ap.parse_args().stage)
