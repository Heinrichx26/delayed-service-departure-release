"""Enumerate all nominal timetables, calculating actual rather than planned checks."""
from collections import defaultdict
from itertools import combinations
from time import perf_counter
import random
from band_design import ROOT,old
from nominal_omission import optimize_omission
from panel_band import save


def brute(problem,lead,delay,budget,width):
    h=problem['horizon'];t0=problem['t0'];paths=problem['paths']
    options=[(r,j) for r in range(t0,h-delay) for j in range(r+delay,min(h,r+lead+1))]
    best=None
    for size in range(min(h,len(options))+1):
        for selected in combinations(options,size):
            if len({j for r,j in selected})<size:continue
            if any(sum(r==t for r,j in selected)>budget for t in range(h)):continue
            releases=[list(problem['prefix']) for _ in paths]
            count=0
            for t in range(t0,h):
                groups=defaultdict(list)
                arrived=[j for r,j in selected if r+delay<=t]
                for s,path in enumerate(paths):
                    key=(tuple(path[:t]),tuple(path[j]//width for j in arrived))
                    groups[key].append(s)
                for members in groups.values():
                    lower=max(problem['bounds'][s][t] for s in members)
                    for s in members:
                        prior=releases[s][-1] if releases[s] else 0
                        releases[s].append(max(prior,lower))
                    for r,j in selected:
                        if r==t and len({paths[s][j]//width for s in members})>1:
                            count+=sum(problem['weights'][s] for s in members)
            waiting=sum(w*sum(u-d for u,d in zip(releases[s][t0:],
                        problem['references'][s][t0:])) for s,w in enumerate(problem['weights']))
            pair=(waiting,count)
            if best is None or pair<best:best=pair
    return best


def main():
    rng=random.Random(2026102847);rows=[];start=perf_counter()
    for i in range(12):
        paths=[]
        for _ in range(3):
            p=[0]*6
            for _ in range(3):p[rng.randrange(3,6)]+=1
            paths.append(p)
        c=dict(n=3,lag=1,caps=[1]*6,service=paths,
               probabilities=['1/6','1/3','1/2'],delta=2)
        for prefix in [(),(1,2)] if i<4 else [()]:
            problem=old.prepare(c,prefix=prefix)
            for d in [1,2]:
                for b in [1,2]:
                    for w in [1,2,4]:
                        answer=optimize_omission(problem,3,d,b,w)
                        oracle=brute(problem,3,d,b,w)
                        assert answer['optimal'] and (
                            answer['weighted_waiting'],answer['weighted_queries'])==oracle
                        rows.append(dict(case=i,prefix=len(prefix),delay=d,budget=b,width=w,value=oracle))
    output=dict(passed=True,conditions=len(rows),seconds=perf_counter()-start,rows=rows)
    save(ROOT/'results/trb_precision_20261028/omission_theory_smoke.json',output)
    print({k:v for k,v in output.items() if k!='rows'})

if __name__=='__main__':main()
