"""Direct report-vector ambiguity checks against the integer covering result."""
from itertools import combinations,product
import json
from pathlib import Path
import random
from robust_reports import prepare
from deadline_cover import solve,balls_intersect

def run():
    rng=random.Random(30102026)
    conditions=0;schedules=0;records=[]
    while conditions<48:
        n=rng.randrange(1,3);h=rng.randrange(3,6);lag=rng.randrange(2)
        paths=tuple(tuple(rng.randrange(n+1) if t>=lag else 0 for t in range(h)) for _ in range(3))
        try:p=prepare(paths,n,(n,)*h,lag)
        except ValueError:continue
        targets=[]
        for b in p['bounds']:
            v=0;row=[]
            for a in b:v=max(v,a);row.append(v)
            targets.append(row)
        delay=1;lead=min(3,h-1);limit=1;gamma=conditions%2
        choices=[(-1,)+tuple(range(max(0,j-lead),j-delay+1)) for j in range(h)]
        best=None
        for timetable in product(*choices):
            selected=[(r,j) for j,r in enumerate(timetable) if r>=0]
            if any(sum(rr==r for rr,j in selected)>limit for r in range(h)):continue
            schedules+=1
            feasible=True
            for s,q in combinations(range(len(paths)),2):
                for t in range(h):
                    if targets[s][t]==targets[q][t] or paths[s][:t]!=paths[q][:t]:continue
                    reports=[j for r,j in selected if r+delay<=t]
                    a=tuple(paths[s][j] for j in reports);b=tuple(paths[q][j] for j in reports)
                    if balls_intersect(a,b,gamma):feasible=False;break
                if not feasible:break
            if feasible and (best is None or len(selected)<best):best=len(selected)
        computed=solve(p,lead,delay,limit,gamma)
        assert computed['feasible']==(best is not None)
        assert computed['requests']==best
        records.append(dict(n=n,h=h,gamma=gamma,minimum_requests=best))
        conditions+=1
    out=Path(__file__).resolve().parents[2]/'results/trb_error_20261030/cover_audit.json'
    result=dict(passed=True,conditions=conditions,enumerated_schedules=schedules,records=records)
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)

if __name__=='__main__':run()
