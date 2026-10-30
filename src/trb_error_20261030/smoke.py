"""Independent small-instance checks before public-data experimentation."""
from __future__ import annotations

import argparse
from functools import lru_cache
from itertools import combinations, product
import json
from pathlib import Path
import random
from time import perf_counter

from robust_reports import RobustReports, physical, prepare

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/trb_error_20261030"


def exhaustive(problem, lead, delay, limit, gamma, price=0, total_requests=None):
    """Full release enumeration; no backward release bounds or production transitions."""
    paths,n,h,caps,lag=(problem[k] for k in ("paths","n","h","caps","lag"))
    refs=problem["refs"]
    @lru_cache(None)
    def solve(t,history,spent,pending,queried):
        if t==h:
            return 0
        prev=history[-1] if history else 0
        eligible=[j for j in range(t+delay,min(h,t+lead+1)) if j not in queried]
        members=[s for s,e in enumerate(spent) if e>=0]
        best=None
        for z in range(prev,min(n,prev+caps[t])+1):
            released=history+(z,)
            # Direct simulation evaluates every candidate physical release.
            if any(physical(released+(z,)*(h-t-1),paths[s],lag)[t]!=refs[s][t]
                   for s in members):
                continue
            for size in range(min(limit,len(eligible))+1):
                if total_requests is not None and len(queried)+size>total_requests:
                    continue
                for batch in combinations(eligible,size):
                    combined=pending+tuple((t+delay,j) for j in batch)
                    due=tuple(j for d,j in combined if d==t+1)
                    future=tuple(sorted((d,j) for d,j in combined if d>t+1))
                    alphabet=set()
                    for s in members:
                        for y in product(range(n+1),repeat=len(due)):
                            if sum(abs(yv-paths[s][j]) for yv,j in zip(y,due))+spent[s]<=gamma:
                                alphabet.add((paths[s][t],y))
                    worst=0
                    valid=True
                    for observation,y in alphabet:
                        nxt=[]
                        for s,e in enumerate(spent):
                            ne=e+sum(abs(yv-paths[s][j]) for yv,j in zip(y,due))
                            nxt.append(ne if e>=0 and paths[s][t]==observation and ne<=gamma else -1)
                        child=solve(t+1,released,tuple(nxt),future,tuple(sorted(queried+batch)))
                        if child is None:
                            valid=False
                            break
                        ref=next(refs[s][t] for s,e in enumerate(nxt) if e>=0)
                        worst=max(worst,z-ref+child)
                    if valid:
                        value=price*size+worst
                        if best is None or value<best:
                            best=value
        return best
    return solve(0,(),(0,)*len(paths),(),())


def run(stage):
    OUT.mkdir(parents=True,exist_ok=True)
    rng=random.Random(20261030)
    cases=[]
    # The nonzero-count differences permit useful checks despite one unit error.
    paths=((0,0,0,2,0,0,2),(0,0,0,0,0,2,2),(0,0,0,0,2,0,2))
    cases.append(prepare(paths,2,(2,)*7,lag=1))
    count=12 if stage=="smoke" else 96
    while len(cases)<count:
        h=rng.randrange(4,7);n=rng.randrange(1,4);lag=rng.randrange(2)
        caps=tuple(rng.randrange(1,n+1) for _ in range(h))
        paths=tuple(tuple(rng.randrange(n+1) if t>=lag else 0 for t in range(h))
                    for _ in range(rng.randrange(2,4)))
        try:
            cases.append(prepare(paths,n,caps,lag))
        except ValueError:
            pass
    records=[]
    for i,p in enumerate(cases):
        for gamma in (0,1):
            for delay in ((1,) if stage=="smoke" else (1,2)):
                m=RobustReports(p,lead=min(p["h"]-1,3),delay=delay,dispatch_limit=1,
                                error_allowance=gamma,seconds_limit=20)
                value=m.solve(); independent=exhaustive(p,m.lead,delay,1,gamma)
                assert value["value"]==independent,(i,gamma,value,independent)
                check=m.verify()
                no=RobustReports(p,m.lead,delay,0,gamma).solve()["value"]
                greedy=RobustReports(p,m.lead,delay,1,gamma,mode="greedy").solve()["value"]
                assert value["value"]<=min(no,greedy)
                records.append(dict(case=i,gamma=gamma,delay=delay,n=p["n"],
                                    exact=value["value"],enumeration=independent,
                                    no_requests=no,greedy=greedy,checks=check["physical_checks"],
                                    states=value["states"],seconds=value["seconds"]))
    # Error allowances are nested: a larger allowance cannot improve minimax time.
    index={(r["case"],r["delay"],r["gamma"]):r for r in records}
    for r in records:
        if r["gamma"]==0:
            assert r["exact"]<=index[r["case"],r["delay"],1]["exact"]
    result=dict(passed=True,stage=stage,conditions=len(records),
                strict_vs_none=sum(r["exact"]<r["no_requests"] for r in records),
                strict_vs_greedy=sum(r["exact"]<r["greedy"] for r in records),
                physical_checks=sum(r["checks"] for r in records),records=records)
    (OUT/(stage+".json")).write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="records"}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["smoke","audit"])
    run(p.parse_args().stage)
