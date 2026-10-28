"""Independent checks of fixed-bin confirmations with exact natural observations."""
from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import interval_theory_validation as theory
import selective_interval_certificate_validation as exact

OUT=theory.OUT
SOURCE=Path(__file__)


def digest():
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def save(name,value):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def report(value,width,n):
    lower=width*(value//width)
    return lower,min(n,lower+width-1)


def intersect(boxes,j,response,n,caps,lag):
    new=[]
    for lo,hi in boxes:
        lower,upper=list(lo),list(hi)
        lower[j]=max(lower[j],response[0])
        upper[j]=min(upper[j],response[1])
        if lower[j]<=upper[j]:
            box=tuple(lower),tuple(upper)
            if exact.complete_upper(box,n,caps,lag):
                new.append(box)
    return tuple(new)


def run(boxes,n,caps,lag,lead,width,actual,selective,prefix=(),reverse=False):
    h,t0=len(caps),len(prefix)
    active=exact.trim(boxes,n,caps,lag)
    reports=set()
    for j in range(t0):
        active=intersect(active,j,(actual[j],actual[j]),n,caps,lag)
    previous=prefix[-1] if prefix else 0
    cumulative=list(prefix)
    queries,checks,coarse_to_exact=0,0,0
    for t in range(t0,h):
        if t:
            # Natural observations always refine a previous coarse response.
            coarse_to_exact+=t-1 in reports and width>1
            active=intersect(active,t-1,(actual[t-1],actual[t-1]),n,caps,lag)
        assert active
        unknown={j for j in range(t,min(h,t+lead+1)) if j not in reports}
        while unknown and previous<n:
            lower,upper,_=exact.bounds(active,unknown,t,previous,n,caps,lag)
            for possible in exact.support(active,n,caps,lag):
                full=active
                for j in unknown:
                    full=intersect(full,j,report(possible[j],width,n),n,caps,lag)
                assert full
                action=max(previous,max(theory.deadlines(n,caps,lag,hi)[1][t]
                                        for _,hi in full))
                assert lower<=action<=upper
                checks+=1
            if selective and lower==upper:
                break
            j=max(unknown) if reverse else min(unknown)
            active=intersect(active,j,report(actual[j],width,n),n,caps,lag)
            assert active
            reports.add(j)
            queries+=1
            unknown.remove(j)
        previous=max(previous,max(theory.deadlines(n,caps,lag,hi)[1][t]
                                  for _,hi in active))
        cumulative.append(previous)
    assert theory.feasible_release(cumulative,n,caps)
    done,inventory=theory.physical(cumulative,actual,lag)
    assert done==theory.deadlines(n,caps,lag,actual)[0]
    assert queries<=h-t0
    return dict(cumulative=cumulative,completed=done,waiting=sum(inventory),
                queries=queries,response_checks=checks,coarse_to_exact=coarse_to_exact)


def check(boxes,n,caps,lag,lead,width,prefix=()):
    totals=dict(paths=0,response_checks=0,coarse_to_exact=0)
    for actual in exact.support(boxes,n,caps,lag):
        full=run(boxes,n,caps,lag,lead,width,actual,False,prefix)
        for reverse in (False,True):
            selected=run(boxes,n,caps,lag,lead,width,actual,True,prefix,reverse)
            assert full['cumulative']==selected['cumulative']
            assert full['completed']==selected['completed']
            assert full['waiting']==selected['waiting']
            totals['paths']+=1
            totals['response_checks']+=selected['response_checks']
            totals['coarse_to_exact']+=selected['coarse_to_exact']+full['coarse_to_exact']
    return totals


def smoke():
    boxes=(((0,0,0,0),(0,2,2,2)),)
    rows=[]
    for width in (1,2,4):
        result=check(boxes,2,(1,1,1,1),1,1,width)
        rows.append(dict(width=width,**result))
    assert rows[1]['coarse_to_exact']>0 and rows[2]['coarse_to_exact']>0
    value=dict(passed=True,source_sha256=digest(),rows=rows)
    save('quantized_interval_smoke.json',value)
    print(value,flush=True)


def exhaustive():
    smoke_result=json.loads((OUT/'quantized_interval_smoke.json').read_text(encoding='utf-8'))
    assert smoke_result['passed'] and smoke_result['source_sha256']==digest()
    begin=perf_counter()
    totals=dict(paths=0,response_checks=0,coarse_to_exact=0)
    cases=0
    h,n=3,2
    # Every centre generates one clipped radius-one box. Pair it with a
    # complementary centre to exercise unions without reusing the main controller.
    for centre in product(range(n+1),repeat=h):
        opposite=tuple(n-x for x in centre)
        boxes=tuple((tuple(max(0,x-1) for x in c),tuple(min(n,x+1) for x in c))
                    for c in (centre,opposite))
        for caps,lag in (((1,1,1),0),((1,1,1),1),((2,0,2),1)):
            paths=exact.support(boxes,n,caps,lag)
            if not paths:
                continue
            prefix=(max(theory.deadlines(n,caps,lag,c)[1][0] for c in paths),)
            for executed in ((),prefix):
                for lead in (-1,0,1,2):
                    for width in (1,2,4):
                        result=check(boxes,n,caps,lag,lead,width,executed)
                        cases+=1
                        for key,value in result.items():
                            totals[key]+=value
    value=dict(passed=True,source_sha256=digest(),cases=cases,**totals,
               elapsed_seconds=perf_counter()-begin)
    save('quantized_interval_exhaustive.json',value)
    print(value,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['smoke','exhaustive'],required=True)
    args=parser.parse_args()
    globals()[args.phase]()
