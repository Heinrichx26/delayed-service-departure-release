"""Compare complete trade-offs to enumeration of physical release actions."""
from collections import defaultdict
from functools import lru_cache
from itertools import combinations
import json
import random
from time import perf_counter
from band_design import BandDesign, ROOT, old
from frontier_design import FrontierDesign
from validate_band import physical, replay


def exhaustive(problem, lead, delay, budget, width):
    paths,h,n=problem['paths'],problem['horizon'],problem['n']
    def nondominated(points):
        return tuple(sorted(p for p in points if not any(
            q!=p and q[0]<=p[0] and q[1]<=p[1] for q in points)))
    @lru_cache(None)
    def step(t,members,pending,prefix):
        if t==h:
            return ((0,0),) if all(physical(prefix,paths[s],problem['lag'])==
                                  problem['references'][s] for s in members) else ()
        prior=prefix[-1] if prefix else 0
        options=[j for j in range(t+delay,min(h,t+lead+1))
                 if j not in {j for _,j in pending}]
        points=set()
        for u in range(prior,min(n,prior+problem['caps'][t])+1):
            ext=prefix+(u,)
            if any(physical(ext,paths[s],problem['lag'])!=
                   problem['references'][s][:t+1] for s in members): continue
            for size in range(min(budget,len(options))+1):
                for request in combinations(options,size):
                    combined=pending+tuple((t+delay,j) for j in request)
                    due=[j for a,j in combined if a==t+1]
                    future=tuple(sorted((a,j) for a,j in combined if a>t+1))
                    groups=defaultdict(list)
                    for s in members:
                        key=(paths[s][t],)+tuple(paths[s][j]//width for j in due)
                        groups[key].append(s)
                    partial={(sum(problem['weights'][s]*(u-problem['references'][s][t])
                                  for s in members),
                              size*sum(problem['weights'][s] for s in members))}
                    for group in groups.values():
                        child=step(t+1,tuple(group),future,ext)
                        partial={(a+c,b+d) for a,b in partial for c,d in child}
                        partial=set(nondominated(partial))
                    points.update(partial)
        return nondominated(points)
    groups=defaultdict(list)
    for s,path in enumerate(paths): groups[tuple(path[:problem['t0']])].append(s)
    total={(0,0)}
    for members in groups.values():
        child=step(problem['t0'],tuple(members),(),tuple(problem['prefix']))
        total={(a+c,b+d) for a,b in total for c,d in child}
        total=set(nondominated(total))
    return tuple(sorted(total))


def main():
    rng=random.Random(2026102837);rows=[];start=perf_counter()
    for i in range(16):
        h=6;n=rng.choice([2,3]);m=3;paths=[]
        for _ in range(m):
            p=[0]*h
            for _ in range(n):p[rng.randrange(2,h)]+=1
            paths.append(p)
        case=dict(n=n,lag=1,caps=[1]*h,service=paths,
                  probabilities=['1/6','1/3','1/2'],delta=2)
        for prefix in [(),(1,2)] if i<8 else [()]:
            problem=old.prepare(case,prefix=prefix)
            for d,b in [(1,1),(1,2),(2,1),(2,2)]:
                for w in [1,2]:
                    model=FrontierDesign(problem,3,d,b,w)
                    answer=model.solve()
                    points=tuple((v['weighted_waiting'],v['weighted_queries'])
                                 for v in answer['frontier'])
                    oracle=exhaustive(problem,3,d,b,w)
                    assert points==oracle,(i,d,b,w,points,oracle)
                    lex=BandDesign(problem,3,d,b,w).solve()
                    assert min(points)==(lex['weighted_waiting'],lex['weighted_queries'])
                    for pair in points:
                        trajectories=model.replay_pair(pair)
                        replay(problem,trajectories,3,d,b,w)
                    rows.append(dict(case=i,prefix=len(prefix),delay=d,budget=b,width=w,points=points))
    result=dict(passed=True,conditions=len(rows),seconds=perf_counter()-start,rows=rows)
    (ROOT/'results/trb_precision_20261028/frontier_smoke.json').write_text(
        json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))

if __name__=='__main__':main()
