"""Independent finite checks of selective interval-confirmation certificates."""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations_with_replacement, product
import json
from pathlib import Path
from time import perf_counter

import interval_theory_validation as theory
import selective_interval as root_selector

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/revision_20261028/substantive_strengthening'
SOURCE = Path(__file__)


def digest():
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def save(name, value):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def complete_upper(box,n,caps,lag):
    return theory.deadlines(n,caps,lag,box[1])[0][-1] == n


def trim(boxes,n,caps,lag):
    return tuple(box for box in boxes if complete_upper(box,n,caps,lag))


def intersect(boxes,j,value,n,caps,lag):
    result=[]
    for lo,hi in boxes:
        if not lo[j] <= value <= hi[j]:
            continue
        new_lo,new_hi=list(lo),list(hi)
        new_lo[j]=new_hi[j]=value
        box=(tuple(new_lo),tuple(new_hi))
        if complete_upper(box,n,caps,lag):
            result.append(box)
    return tuple(result)


def support(boxes,n,caps,lag):
    calendars=set()
    for lo,hi in boxes:
        calendars.update(theory.clipped_box_paths(lo,hi,n,caps,lag))
    return tuple(sorted(calendars))


def bounds(boxes,unknown,t,previous,n,caps,lag):
    upper=max(previous,max(theory.deadlines(n,caps,lag,hi)[1][t]
                           for _,hi in boxes))
    lowers=[]
    incomplete_mixed=0
    for lo,hi in boxes:
        mixed=tuple(lo[j] if j in unknown else hi[j] for j in range(len(caps)))
        incomplete_mixed += not theory.deadlines(n,caps,lag,mixed)[0][-1] == n
        lowers.append(theory.deadlines(n,caps,lag,mixed)[1][t])
    lower=max(previous,min(lowers))
    assert lower <= upper
    return lower,upper,incomplete_mixed


def run(boxes,n,caps,lag,lead,actual,selective,prefix=(),reverse=False,
        root_order=False,infer_constants=False):
    h,t0=len(caps),len(prefix)
    active=trim(boxes,n,caps,lag)
    known={j:actual[j] for j in range(t0)}
    for j,value in known.items():
        active=intersect(active,j,value,n,caps,lag)
    cumulative=list(prefix)
    previous=prefix[-1] if prefix else 0
    queries,fallback,certificates,mixed=0,None,0,0
    stopped=0
    for t in range(t0,h):
        if fallback is None:
            if t and t-1 not in known:
                known[t-1]=actual[t-1]
                active=intersect(active,t-1,actual[t-1],n,caps,lag)
            if not active:
                fallback=t
        if fallback is None:
            unknown={j for j in range(t,min(h,t+lead+1)) if j not in known}
            if previous == n:
                unknown=set()
            while unknown:
                if infer_constants:
                    unknown={j for j in unknown
                             if min(lo[j] for lo,hi in active) < max(hi[j] for lo,hi in active)}
                    if not unknown:
                        break
                lower,upper,invalid=bounds(active,unknown,t,previous,n,caps,lag)
                mixed+=invalid
                # Explicitly enumerate every feasible response calendar at this node.
                for possible in support(active,n,caps,lag):
                    full=active
                    for j in unknown:
                        full=intersect(full,j,possible[j],n,caps,lag)
                    assert full
                    full_action=max(previous,max(theory.deadlines(n,caps,lag,hi)[1][t]
                                                 for _,hi in full))
                    assert lower <= full_action <= upper
                    certificates+=1
                if selective and lower == upper:
                    stopped+=1
                    break
                if root_order:
                    as_lists=[(list(lo),list(hi)) for lo,hi in active]
                    j,certificate=root_selector.choose_confirmation(
                        as_lists,n,caps,lag,t,previous,lead,known)
                    assert certificate['upper_release']==upper
                    assert certificate['lower_release']==lower
                    if j is None:
                        assert lower==upper or all(
                            min(lo[k] for lo,hi in active)==max(hi[k] for lo,hi in active)
                            for k in unknown)
                        stopped+=1
                        break
                    assert j in unknown
                else:
                    j=max(unknown) if reverse else min(unknown)
                known[j]=actual[j]
                queries+=1
                unknown.remove(j)
                active=intersect(active,j,actual[j],n,caps,lag)
                if not active:
                    fallback=t
                    break
        if fallback is None:
            previous=max(previous,max(theory.deadlines(n,caps,lag,hi)[1][t]
                                      for _,hi in active))
        else:
            previous=min(n,previous+caps[t])
        cumulative.append(previous)
    assert theory.feasible_release(cumulative,n,caps)
    completed,inventory=theory.physical(cumulative,actual,lag)
    reference=theory.deadlines(n,caps,lag,actual)[0]
    return dict(cumulative=cumulative,queries=queries,fallback=fallback,
                completed=completed,waiting=sum(inventory),
                max_deficit=max(a-d for a,d in zip(reference,completed)),
                certificate_response_checks=certificates,
                incomplete_mixed_lower_checks=mixed,early_stops=stopped)


def check_case(boxes,n,caps,lag,lead,prefix=()):
    paths=support(boxes,n,caps,lag)
    totals=dict(paths=0,certificate_response_checks=0,
                incomplete_mixed_lower_checks=0,strict_query_savings=0,
                larger_query_counts=0,early_stops=0)
    for actual in paths:
        full=run(boxes,n,caps,lag,lead,actual,False,prefix,infer_constants=True)
        for order in ('ascending','descending','root'):
            selected=run(boxes,n,caps,lag,lead,actual,True,prefix,
                         reverse=order=='descending',root_order=order=='root',infer_constants=True)
            assert selected['fallback'] is full['fallback'] is None
            assert selected['cumulative'] == full['cumulative']
            assert selected['completed'] == full['completed']
            assert selected['waiting'] == full['waiting']
            assert selected['max_deficit'] == full['max_deficit'] == 0
            assert selected['queries'] <= len(caps)-len(prefix)
            totals['paths']+=1
            totals['certificate_response_checks']+=selected['certificate_response_checks']
            totals['incomplete_mixed_lower_checks']+=selected['incomplete_mixed_lower_checks']
            totals['strict_query_savings']+=selected['queries'] < full['queries']
            totals['larger_query_counts']+=selected['queries'] > full['queries']
            totals['early_stops']+=selected['early_stops']
    return totals


def smoke():
    n,caps,lag,lead=1,(1,1,1,1,1),1,3
    nominal=(0,0,0,0,1)
    boxes=((nominal,nominal),)
    covered=check_case(boxes,n,caps,lag,lead)
    actual=(0,1,0,0,0)
    full=run(boxes,n,caps,lag,lead,actual,False)
    selected=run(boxes,n,caps,lag,lead,actual,True)
    assert full['fallback']==0 and selected['fallback']==2
    assert full['max_deficit']==0 and selected['max_deficit']==1
    assert full['cumulative'] != selected['cumulative']
    interval=check_case((((0,0,0,0),(0,2,2,2)),),2,(1,1,1,1),1,1)
    assert interval['incomplete_mixed_lower_checks'] > 0
    fair_boxes=(((0,0,0,3),(0,1,0,3)),)
    fair_actual=(0,3,0,3)
    fair_full=run(fair_boxes,3,(3,0,1,3),1,1,fair_actual,False,infer_constants=True)
    fair_selected=run(fair_boxes,3,(3,0,1,3),1,1,fair_actual,True,root_order=True,infer_constants=True)
    assert fair_full['fallback']==0 and fair_selected['fallback']==2
    assert fair_full['max_deficit']==0 and fair_selected['max_deficit']==1
    report=dict(passed=True,source_sha256=digest(),
                root_selector_sha256=hashlib.sha256(Path(root_selector.__file__).read_bytes()).hexdigest(),
                covered_point=covered,
                covered_interval=interval,out_of_support_negative_control=dict(
                    n=n,caps=caps,lag=lag,lead=lead,nominal=nominal,actual=actual,
                    full=full,selective=selected),
                out_of_support_fair_baseline_negative_control=dict(
                    boxes=fair_boxes,actual=fair_actual,n=3,caps=(3,0,1,3),lag=1,lead=1,
                    full=fair_full,selective=fair_selected))
    save('selective_interval_smoke.json',report)
    print(dict(passed=True,covered_point=covered,covered_interval=interval,
               outside_full_fallback=full['fallback'],outside_selective_fallback=selected['fallback']),flush=True)


def exhaustive():
    smoke_report=json.loads((OUT/'selective_interval_smoke.json').read_text(encoding='utf-8'))
    assert smoke_report['passed'] and smoke_report['source_sha256']==digest()
    begin=perf_counter()
    totals=dict(paths=0,certificate_response_checks=0,
                incomplete_mixed_lower_checks=0,strict_query_savings=0,
                larger_query_counts=0,early_stops=0)
    cases,prefix_cases=0,0
    h=3
    all_boxes=[]
    for coordinates in product(((0,0),(0,1),(1,1)),repeat=h):
        all_boxes.append((tuple(x[0] for x in coordinates),tuple(x[1] for x in coordinates)))
    for pair in combinations_with_replacement(all_boxes,2):
        for n,caps,lag in ((1,(1,1,1),0),(1,(1,1,1),1),
                           (2,(1,1,1),0),(2,(2,0,2),1)):
            paths=support(pair,n,caps,lag)
            if not paths:
                continue
            prefix=(max(theory.deadlines(n,caps,lag,c)[1][0] for c in paths),)
            assert prefix[0] <= caps[0]
            for executed in ((),prefix):
                for lead in (-1,0,1,2):
                    result=check_case(pair,n,caps,lag,lead,executed)
                    cases+=1
                    prefix_cases+=bool(executed)
                    for key,value in result.items():
                        totals[key]+=value
    report=dict(passed=True,source_sha256=digest(),cases=cases,
                root_selector_sha256=hashlib.sha256(Path(root_selector.__file__).read_bytes()).hexdigest(),
                nonempty_prefix_cases=prefix_cases,**totals,
                elapsed_seconds=perf_counter()-begin)
    save('selective_interval_exhaustive.json',report)
    print(report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['smoke','exhaustive'],required=True)
    args=parser.parse_args()
    globals()[args.phase]()
