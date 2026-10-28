"""Small independent enumerations precede every public-data experiment."""
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
from itertools import combinations
import json
from pathlib import Path
import random
from time import perf_counter
from delayed_acquisition import DelayedDesign, delayed_targets, fixed_schedule, old, ROOT

def raw_value(problem,lead,delay,budget,target):
    paths=problem['paths'];m=len(paths);h=problem['horizon']
    @lru_cache(None)
    def solve(t,members,pending):
        if t==h:return 0
        if len({target[s][t] for s in members})!=1:return None
        choices=[j for j in range(t+delay,min(h,t+lead+1))
                 if j not in {q[1] for q in pending} and len({paths[s][j] for s in members})>1]
        best=None
        for size in range(min(budget,len(choices))+1):
            for query in combinations(choices,size):
                allpending=pending+tuple((t+delay,j) for j in query)
                arriving=[j for due,j in allpending if due==t+1]
                remaining=tuple(sorted((due,j) for due,j in allpending if due>t+1))
                children=defaultdict(list)
                for s in members:children[tuple([paths[s][t]]+[paths[s][j] for j in arriving])].append(s)
                cost=size*sum(problem['weights'][s] for s in members)
                good=True
                for child in children.values():
                    result=solve(t+1,tuple(child),remaining)
                    if result is None:good=False;break
                    cost+=result
                if good and (best is None or cost<best):best=cost
        return best
    value=solve(0,tuple(range(m)),())
    return value,solve.cache_info().currsize

def independent_replay(problem,target,replays,lead,delay,budget):
    paths=problem['paths'];weighted=0
    for row in replays:
        s=row['s'];queries=row['queries'];path=paths[s];release=row['release']
        for q in queries:
            assert q['arrival']==q['dispatch']+delay
            assert max(problem['t0'],q['slot']-lead)<=q['dispatch']
        for t in range(problem['horizon']):
            assert sum(q['dispatch']==t for q in queries)<=budget
            information=list(range(t))+[q['slot'] for q in queries if q['arrival']<=t]
            compatible=[k for k,p in enumerate(paths) if all(p[j]==path[j] for j in information)]
            assert len({target[k][t] for k in compatible})==1
            assert release[t]==target[s][t]
        done=0;completed=[];previous=0
        for t,total in enumerate(release):
            assert 0<=total-previous<=problem['caps'][t];previous=total
            available=release[t-problem['lag']] if t>=problem['lag'] else 0
            done+=min(path[t],available-done);completed.append(done)
        assert completed==problem['references'][s]
        after=problem['case']['delta']*sum(u-d for u,d in zip(release,completed))/problem['n']
        assert abs(after-row['post_release'])<1e-9
        weighted+=problem['weights'][s]*len(queries)
    return weighted

def brute_fixed(problem,lead,delay,budget,target):
    paths=problem['paths'];h=problem['horizon']
    options=[(t,j) for t in range(h-delay) for j in range(t+delay,min(h,t+lead+1))
             if len({p[j] for p in paths})>1]
    for size in range(min(h,len(options))+1):
        for chosen in combinations(options,size):
            if len({j for _,j in chosen})<size:continue
            if any(sum(t==v for t,_ in chosen)>budget for v in range(h)):continue
            good=True
            for t in range(h):
                observed=list(range(t))+[j for v,j in chosen if v+delay<=t]
                groups=defaultdict(set)
                for s,p in enumerate(paths):groups[tuple(p[j] for j in observed)].add(target[s][t])
                if any(len(v)>1 for v in groups.values()):good=False;break
            if good:return size
    return None

def main():
    start=perf_counter();rng=random.Random(2026102809);rows=[]
    for index in range(48):
        h=6;n=rng.choice([1,2]);paths=[]
        for s in range(rng.choice([2,3,4])):
            p=[0]*h
            for _ in range(n):p[rng.randrange(2,h)]+=1
            paths.append(p)
        case=dict(n=n,lag=1,caps=[1]*h,service=paths,probabilities=[f'1/{len(paths)}']*len(paths),delta=2)
        problem=old.prepare(case)
        assert problem is not None
        for delay in [1,2]:
            for budget in [1,2]:
                lead=3;target=delayed_targets(problem,lead,delay)
                model=DelayedDesign(problem,lead,delay,budget,target,state_limit=200000)
                answer=model.solve();raw,rawstates=raw_value(problem,lead,delay,budget,target)
                assert answer['weighted_cost']==raw,(index,delay,budget,answer,raw)
                if answer['feasible']:
                    assert independent_replay(problem,target,model.replay(),lead,delay,budget)==raw
                fixed=None
                if index<10:
                    fixed=fixed_schedule(problem,lead,delay,budget,target)
                    exhaustive=brute_fixed(problem,lead,delay,budget,target)
                    assert fixed['optimal'] and fixed['feasible']==(exhaustive is not None)
                    if exhaustive is not None:
                        assert fixed['count']==exhaustive
                        assert answer['feasible'] and answer['expected_queries']<=exhaustive+1e-9
                rows.append(dict(index=index,delay=delay,budget=budget,weighted_cost=raw,
                                 optimized_states=answer['states'],raw_states=rawstates,fixed=fixed))
    case=dict(n=1,lag=1,caps=[1]*6,service=[[0,0,0,1,0,0],[0,0,0,0,0,1]],probabilities=['1/2','1/2'],delta=2)
    problem=old.prepare(case);model=DelayedDesign(problem,3,1,1)
    answer=model.solve();replay=model.replay()
    assert answer['expected_queries']==1
    assert all(q['dispatch']<2 for r in replay for q in r['queries'])
    assert model.target[0][1]==model.target[1][1]==0
    assert model.target[0][2]!=model.target[1][2]
    result=dict(passed=True,seed=2026102809,exact_policy_comparisons=len(rows),
                exact_schedule_comparisons=sum(r['fixed'] is not None for r in rows),
                counterexample=dict(case=case,answer=answer,replay=replay,target=model.target),
                seconds=perf_counter()-start,rows=rows)
    out=ROOT/'results/trb_latency_20261028/theory_smoke.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','counterexample']}))

if __name__=='__main__':main()
