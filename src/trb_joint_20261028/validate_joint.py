"""Independent full-action enumeration and nonanticipativity replay."""
from collections import defaultdict
from functools import lru_cache
from itertools import combinations
import json
import random
from time import perf_counter
from joint_design import JointDesign,ROOT,old

def physical(prefix,path,lag):
    completed=0; previous=0; result=[]
    for t,u in enumerate(prefix):
        assert u>=previous
        available=prefix[t-lag] if t>=lag else 0
        completed=min(available,completed+path[t])
        result.append(completed); previous=u
    return result

def raw(problem,lead,delay,budget):
    paths=problem['paths'];h=problem['horizon'];n=problem['n'];weights=problem['weights']
    @lru_cache(None)
    def recurse(t,members,pending,prefix):
        if t==h:
            return (0,0) if all(physical(prefix,paths[s],problem['lag'])==problem['references'][s] for s in members) else None
        candidates=[j for j in range(t+delay,min(h,t+lead+1)) if j not in {j for _,j in pending}
                    and len({paths[s][j] for s in members})>1]
        best=None;previous=prefix[-1] if prefix else 0
        for release in range(previous,min(n,previous+problem['caps'][t])+1):
            extended=prefix+(release,)
            if any(physical(extended,paths[s],problem['lag'])!=problem['references'][s][:t+1] for s in members):
                continue
            stage=sum(weights[s]*(release-problem['references'][s][t]) for s in members)
            for size in range(min(budget,len(candidates))+1):
                for batch in combinations(candidates,size):
                    allpending=pending+tuple((t+delay,j) for j in batch)
                    due=[j for d,j in allpending if d==t+1]
                    future=tuple(sorted((d,j) for d,j in allpending if d>t+1))
                    groups=defaultdict(list)
                    for s in members:
                        groups[tuple([paths[s][t]]+[paths[s][j] for j in due])].append(s)
                    value=[stage,size*sum(weights[s] for s in members)]
                    for group in groups.values():
                        child=recurse(t+1,tuple(group),future,extended)
                        if child is None:break
                        value[0]+=child[0];value[1]+=child[1]
                    else:
                        pair=tuple(value)
                        if best is None or pair<best:best=pair
        return best
    initial=defaultdict(list)
    for s,p in enumerate(paths):initial[tuple(p[:problem['t0']])].append(s)
    values=[recurse(problem['t0'],tuple(members),(),tuple(problem['prefix'])) for members in initial.values()]
    value=tuple(sum(v[k] for v in values) for k in (0,1))
    return value,recurse.cache_info().currsize

def replay(problem,rows,lead,delay,budget):
    """Reconstruct each information history and every physical departure."""
    histories={}
    for row in rows:
        s=row['s'];path=problem['paths'][s];queries=row['queries'];release=row['release']
        assert physical(release,path,problem['lag'])==problem['references'][s]
        previous=0
        for t,u in enumerate(release):
            assert 0<=u-previous<=problem['caps'][t];previous=u
            batch=tuple(q['slot'] for q in queries if q['dispatch']==t)
            assert len(batch)<=budget
            for q in queries:
                assert q['arrival']==q['dispatch']+delay
                assert q['slot']-lead<=q['dispatch']<=q['slot']-delay
            arrived=tuple((q['dispatch'],q['slot'],path[q['slot']]) for q in queries if q['arrival']<=t)
            sent=tuple((q['dispatch'],q['slot']) for q in queries if q['dispatch']<t)
            if t<problem['t0']:continue
            key=(t,tuple(path[:t]),arrived,sent)
            action=(u,batch)
            if key in histories:assert histories[key]==action
            histories[key]=action
    return len(histories)

def main():
    start=perf_counter();rng=random.Random(2026102817);rows=[]
    for i in range(36):
        h=6;n=rng.choice([1,2]);m=rng.choice([2,3,4])
        paths=[]
        for _ in range(m):
            p=[0]*h
            for k in range(n):p[rng.randrange(2,h)]+=1
            paths.append(p)
        case=dict(n=n,lag=1,caps=[rng.choice([1,2]) for _ in range(h)],service=paths,
                  probabilities=[f'1/{m}']*m,delta=2)
        problem=old.prepare(case);assert problem is not None
        for d in [1,2]:
            for b in [1,2]:
                model=JointDesign(problem,3,d,b);ans=model.solve()
                oracle,states=raw(problem,3,d,b)
                assert (ans['weighted_waiting'],ans['weighted_queries'])==oracle,(i,d,b,ans,oracle)
                replay(problem,model.replay(),3,d,b)
                rows.append(dict(case=i,delay=d,budget=b,value=oracle,states=ans['states'],raw_states=states))
        if i<12:
            prefix=old.base.earliest(n,case['caps'])[:2]
            reset=old.prepare(case,prefix=prefix);assert reset is not None
            for d in [1,2]:
                for b in [1,2]:
                    model=JointDesign(reset,3,d,b);ans=model.solve()
                    oracle,states=raw(reset,3,d,b)
                    assert (ans['weighted_waiting'],ans['weighted_queries'])==oracle
                    replay(reset,model.replay(),3,d,b)
                    rows.append(dict(case=i,reset=2,delay=d,budget=b,value=oracle,states=ans['states'],raw_states=states))
    out=dict(passed=True,comparisons=len(rows),seed=2026102817,seconds=perf_counter()-start,rows=rows)
    (ROOT/'results/trb_joint_20261028/theory_smoke.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='rows'}))

if __name__=='__main__':main()
