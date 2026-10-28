"""Enumeration of fixed interval-report schedules, including inherited releases."""
from collections import defaultdict
from itertools import combinations
import json
import random
from time import perf_counter
from band_design import BandDesign, ROOT, old
from fixed_band import fixed_band
from unlimited_band import unlimited
from validate_band import replay


def brute(problem, lead, delay, budget, width):
    paths, h, t0 = problem['paths'], problem['horizon'], problem['t0']
    candidates=[(r,j) for r in range(t0,h-delay)
                for j in range(r+delay,min(h,r+lead+1))]
    best=None
    for size in range(min(h,len(candidates))+1):
        for selected in combinations(candidates,size):
            if len({j for _,j in selected}) != size: continue
            if any(sum(r==t for r,j in selected)>budget for t in range(h)): continue
            releases=[list(problem['prefix']) for _ in paths]
            for t in range(t0,h):
                groups=defaultdict(list)
                arrived=[j for r,j in selected if r+delay<=t]
                for s,path in enumerate(paths):
                    key=(tuple(path[:t]),tuple(path[j]//width for j in arrived))
                    groups[key].append(s)
                for members in groups.values():
                    lower=max(problem['bounds'][s][t] for s in members)
                    for s in members:
                        previous=releases[s][-1] if releases[s] else 0
                        releases[s].append(max(previous,lower))
            weighted=sum(w*sum(u-d for u,d in zip(releases[s][t0:],
                         problem['references'][s][t0:])) for s,w in enumerate(problem['weights']))
            pair=(weighted,size)
            if best is None or pair<best: best=pair
    return best


def main():
    rng=random.Random(2026102831);rows=[];start=perf_counter()
    for i in range(8):
        n=3;m=3;paths=[]
        for _ in range(m):
            p=[0]*6
            for _ in range(n):p[rng.randrange(2,6)]+=1
            paths.append(p)
        case=dict(n=n,lag=1,caps=[1]*6,service=paths,probabilities=['1/6','1/3','1/2'],delta=2)
        for prefix in [(),(1,2)]:
            p=old.prepare(case,prefix=prefix)
            for d in [1,2]:
                for b in [1,2]:
                    for w in [1,2,4]:
                        answer=fixed_band(p,3,d,b,w)
                        oracle=brute(p,3,d,b,w)
                        assert answer['optimal'] and (answer['weighted_waiting'],answer['count'])==oracle
                        replay(p,answer['paths'],3,d,b,w)
                        adaptive=BandDesign(p,3,d,b,w).solve()
                        assert adaptive['weighted_waiting']<=answer['weighted_waiting']
                        floor=unlimited(p,3,d,w)
                        all_reports=BandDesign(p,3,d,3,w).solve()
                        assert floor['weighted_waiting']==all_reports['weighted_waiting']
                        rows.append(dict(case=i,prefix=len(prefix),delay=d,budget=b,width=w,value=oracle))
    out=dict(passed=True,conditions=len(rows),rows=rows,seconds=perf_counter()-start)
    (ROOT/'results/trb_precision_20261028/fixed_smoke.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='rows'}))


if __name__=='__main__':main()
