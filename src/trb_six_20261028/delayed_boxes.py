"""Date-separated interval control with delayed reports and finite dispatches."""
from itertools import combinations
from math import ceil
from pathlib import Path
from time import perf_counter
from functools import lru_cache
import sys, json, hashlib
import numpy as np
from extensions import ROOT,read,save,reset_loss
sys.path.insert(0,str(ROOT/'src/trb_independent_20261028'))
from calibration_panel import calibrate, physical

def run_policy(original,split,delay,budget,method='batch',width=1,mode='valid'):
    case,radius,scores=calibrate(original,split)
    n=case['n'];h=len(case['caps']);lag=case['lag'];lead=lag+2
    caps=[ceil(n/4)]*h
    initial=physical.boxes(case,radius)
    actual=case['actual_service']
    # A response is generated from a training-period forecast path, never from
    # the held-out realised service calendar. The latter is used only for
    # evaluation after the policy has completed.
    forecasts=case['service']
    def forecast_for(t,j,revision=0):
        key=f"{case['key']}|{split}|{t}|{j}|{revision}".encode('utf-8')
        seed=int.from_bytes(hashlib.sha256(key).digest()[:8],'big')
        return forecasts[seed % len(forecasts)][j]
    covered=any(all(a<=v<=b for a,v,b in zip(lo,actual,hi)) for lo,hi in initial)
    observed={};reports=[];pending=[];queries=[];prefix=[];snapshots=[];timing=[]
    prior=0;incompatible=0;infeasible_resets=0;maximum_unavoidable=0;corrections=0
    bound_calls=0
    @lru_cache(maxsize=50000)
    def bounds(service):
        nonlocal bound_calls
        bound_calls+=1
        ref=physical.reference(case,caps,service)
        b=physical.deadline_bound(ref,caps,lag)
        # Release the complete group by the final decision even if service is low.
        b[-1]=n
        for j in range(h-2,-1,-1):b[j]=max(b[j],b[j+1]-caps[j+1])
        return tuple(b)
    def active_at(t,extra=()):
        valid=[r for r in reports if r['arrival']<=t<r['expires']]
        boxes=[]
        for lo,hi in initial:
            a=list(lo);b=list(hi)
            for j,v in observed.items():a[j]=max(a[j],v);b[j]=min(b[j],v)
            for r in valid+list(extra):
                j=r['slot'];a[j]=max(a[j],r['lo']);b[j]=min(b[j],r['hi'])
            if all(x<=y for x,y in zip(a,b)):boxes.append((a,b))
        return boxes
    def current_bound(active):
        return [max(values) for values in zip(*(bounds(tuple(hi)) for lo,hi in active))]
    def continuation_cost(active,arrival):
        if not active:return None
        b=current_bound(active);previous=prior;total=0
        for k in range(t,h):
            previous=max(previous,b[k])
            if k>=arrival:total+=previous
        return total
    def report_interval(value):
        low=value//width*width
        return low,min(n,low+width-1)
    for t in range(h):
        tick=perf_counter()
        if t:observed[t-1]=actual[t-1]
        for r in pending:
            if r['arrival']==t:
                reports.append(r)
                corrections+=int(r.get('correction',False))
        pending=[r for r in pending if r['arrival']>t]
        active=active_at(t)
        batch=()
        if not active:
            incompatible+=1
            action=min(n,prior+caps[t])
        else:
            upper=current_bound(active)
            # Expansion can invalidate already executed releases. Check full history.
            feasibility=all(all(prefix[k]>=bounds(tuple(hi))[k] for k in range(t))
                            for lo,hi in active)
            if not feasibility:
                infeasible_resets+=1
                loss=reset_loss(case,caps,prefix,paths=[hi for lo,hi in active])
                maximum_unavoidable=max(maximum_unavoidable,max(r['maximum'] for r in loss['paths']))
            action=min(n,prior+caps[t]) if not feasibility else max(prior,upper[t])
            assert prior<=action<=min(n,prior+caps[t])
            existing={r['slot'] for r in pending if not r.get('correction',False)}
            valid_slots={r['slot'] for r in reports if r['arrival']<=t<r['expires']}
            candidates=[j for j in range(t+delay,min(h,t+lead+1))
                        if j not in existing and j not in valid_slots and
                        max(hi[j] for lo,hi in active)>min(lo[j] for lo,hi in active)]
            if prior<n and candidates and method!='none':
                if method=='chronological':
                    batch=tuple(candidates[:budget])
                elif method=='widest':
                    batch=tuple(sorted(candidates,key=lambda j:
                        (-(max(hi[j] for lo,hi in active)-min(lo[j] for lo,hi in active)),j))[:budget])
                elif method in ('batch','robust'):
                    arrival=t+delay
                    # Equal-weight earlier-date centre counts approximating report outcomes.
                    # Prospective reports alone are added; no future realized count is read.
                    representatives=[]
                    for centre in case['service']:
                        distances=[sum(max(a-v,0,v-b) for a,v,b in zip(lo,centre,hi)) for lo,hi in active]
                        lo,hi=active[min(range(len(active)),key=lambda i:distances[i])]
                        representatives.append([min(b,max(a,v)) for a,v,b in zip(lo,centre,hi)])
                    baseline=continuation_cost(active,arrival)
                    best=(baseline,0,())
                    for size in range(1,min(budget,len(candidates))+1):
                        for candidate in combinations(candidates,size):
                            outcomes={}
                            for representative in representatives:
                                key=tuple(report_interval(representative[j]) for j in candidate)
                                outcomes[key]=outcomes.get(key,0)+1
                            total=0; outcome_values=[]
                            for outcome,weight in outcomes.items():
                                extra=[dict(slot=j,lo=lo,hi=hi) for j,(lo,hi) in zip(candidate,outcome)]
                                # Intersect against current boxes, preserving all current evidence.
                                narrowed=[]
                                for lo,hi in active:
                                    a=list(lo);b=list(hi)
                                    for r in extra:
                                        j=r['slot'];a[j]=max(a[j],r['lo']);b[j]=min(b[j],r['hi'])
                                    if all(x<=y for x,y in zip(a,b)):narrowed.append((a,b))
                                value=continuation_cost(narrowed,arrival)
                                assert value is not None
                                total+=weight*value
                                outcome_values.append(value)
                            if method=='robust':
                                # Minimize the worst continuation across the
                                # forecast representatives. This protects the
                                # release decision from an optimistic report.
                                score=(max(outcome_values),size,candidate)
                            else:
                                score=(total/len(representatives),size,candidate)
                            if score<best:best=score
                    batch=best[2]
                else:raise ValueError(method)
            for j in batch:
                # Advance observations are ideal-valid by construction in this family.
                # Understatement, expiry and corrections are explicit stress settings.
                value=forecast_for(t,j)
                reported=max(0,value-1) if mode in ('understate','corrected') else value
                low,high=report_interval(reported)
                arrival=t+delay
                expires=min(h,arrival+1) if mode=='expire' else h
                pending.append(dict(dispatch=t,arrival=arrival,slot=j,lo=low,hi=high,expires=expires))
                queries.append(dict(dispatch=t,arrival=arrival,slot=j,lo=low,hi=high,expires=expires))
                if mode=='corrected' and arrival+1<h:
                    corrected=forecast_for(t,j,1)
                    low,high=report_interval(corrected)
                    pending.append(dict(dispatch=t,arrival=arrival+1,slot=j,lo=low,hi=high,
                                        expires=h,correction=True))
        assert len(batch)<=budget
        prefix.append(action);prior=action
        timing.append(perf_counter()-tick)
        snapshots.append(dict(t=t,release=action,request=list(batch),compatible_boxes=len(active)))
    result=physical.evaluate(case,caps,prefix,None,len(queries))
    gate=case['delta']*sum(n-u for u in prefix)/n
    total=case['delta']*sum(n-d for d in result['completed'])/n
    assert abs(gate+result['waiting_per_aircraft']-total)<1e-8
    # Held-out realised service is evaluated after the policy is fixed. A
    # forecast report can therefore fail to preserve that realised trajectory;
    # such failures are recorded as the operational cost of forecast error.
    reports_cover_actual=all(q['lo']<=actual[q['slot']]<=q['hi'] for q in queries)
    return dict(split=split,radius=radius,delay=delay,budget=budget,method=method,width=width,
                mode=mode,n=n,covered=covered,reports_cover_actual=reports_cover_actual,
                preserved=result['completion_preserving'],
                complete=result['actual_reference_clears'],deficit=result['maximum_completion_deficit'],
                deficit_minutes=result['completion_deficit_aircraft_minutes'],
                post_release=result['waiting_per_aircraft'],gate_wait=gate,start_to_takeoff=total,
                requests=len(queries),incompatible_decisions=incompatible,
                infeasible_resets=infeasible_resets,maximum_unavoidable=maximum_unavoidable,
                corrections=corrections,timing=timing,queries=queries,releases=prefix,
                completed=result['completed'],snapshots=snapshots,bound_calls=bound_calls)

def smoke():
    manifest=read(ROOT/'data/substantive_20261028/holdout_boxes/panel_design.json')
    selected=[next(r for r in manifest['rows'] if r['airport']==a) for a in ['DEN','PHX']]
    rows=[]
    for row in selected:
        case=read(ROOT/row['input_path'])
        for d,b in [(1,1),(2,2)]:
            for method in ['none','batch','robust','chronological','widest']:
                r=run_policy(case,8,d,b,method)
                rows.append(dict(case_id=row['case_id'],**r))
        for mode in ['expire','understate','corrected']:
            rows.append(dict(case_id=row['case_id'],**run_policy(case,8,1,1,'batch',mode=mode)))
        # Report selection uses no future held-out values.
        changed=dict(case,actual_service=case['actual_service'][:])
        changed['actual_service'][-1]+=99
        a=run_policy(case,8,1,1,'batch')
        b=run_policy(changed,8,1,1,'batch')
        assert a['snapshots'][0]['request']==b['snapshots'][0]['request']
    out=dict(passed=True,rows=rows,windows=2,decisions=sum(len(r['timing']) for r in rows),
             maximum_seconds=max(max(r['timing']) for r in rows))
    save(ROOT/'results/trb_six_20261028/box_smoke.json',out)
    print(json.dumps({k:v for k,v in out.items() if k!='rows'}),flush=True)

def one_row(row):
    case=read(ROOT/row['input_path'])
    out=[]
    # The split and response-resource settings are fixed before evaluation.
    for split in (8,12):
        for delay in (1,2):
            for budget in (1,2):
                for method in ('none','batch','robust','chronological','widest'):
                    out.append(dict(case_id=row['case_id'],airport=row['airport'],day=row['day'],
                                    **run_policy(case,split,delay,budget,method)))
        for mode in ('expire','understate','corrected'):
            out.append(dict(case_id=row['case_id'],airport=row['airport'],day=row['day'],
                            **run_policy(case,split,1,1,'batch',mode=mode)))
    return out

def summarize(rows):
    from statistics import mean
    keys=sorted({(r['split'],r['delay'],r['budget'],r['method']) for r in rows
                 if r['mode']=='valid'})
    ans=[]
    for split,delay,budget,method in keys:
        g=[r for r in rows if r['mode']=='valid' and
           (r['split'],r['delay'],r['budget'],r['method'])==(split,delay,budget,method)]
        ans.append(dict(split=split,delay=delay,budget=budget,method=method,windows=len(g),
                        complete=sum(r['complete'] for r in g),preserved=sum(r['preserved'] for r in g),
                        reports_cover_actual=sum(r['reports_cover_actual'] for r in g),
                        mean_deficit_minutes=mean(r['deficit_minutes'] for r in g),
                        mean_post_release=mean(r['post_release'] for r in g),
                        mean_requests=mean(r['requests'] for r in g),
                        p95_decision_seconds=sorted(x for r in g for x in r['timing'])[max(0,int(.95*sum(len(r['timing']) for r in g))-1)]))
    for mode in ('expire','understate','corrected'):
        g=[r for r in rows if r['mode']==mode]
        ans.append(dict(mode=mode,windows=len(g),complete=sum(r['complete'] for r in g),
                        preserved=sum(r['preserved'] for r in g),mean_deficit_minutes=mean(r['deficit_minutes'] for r in g),
                        max_unavoidable=max(r['maximum_unavoidable'] for r in g)))
    return ans

def run():
    from concurrent.futures import ProcessPoolExecutor
    manifest=read(ROOT/'data/substantive_20261028/holdout_boxes/panel_design.json')
    all_rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,rows in enumerate(pool.map(one_row,manifest['rows']),1):
            all_rows.extend(rows)
            if i%10==0: print(i,flush=True)
    save(ROOT/'results/trb_six_20261028/box_rows.json',all_rows)
    save(ROOT/'results/trb_six_20261028/box_summary.json',summarize(all_rows))
    save(ROOT/'results/trb_six_20261028/box_audit.json',dict(passed=True,windows=94,rows=len(all_rows),workers=2,
          source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(dict(passed=True,windows=94,rows=len(all_rows))),flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='run': run()
    else: smoke()
