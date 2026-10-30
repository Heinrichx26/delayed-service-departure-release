"""Exact fixed request schedule for an error tolerant full information target."""
from itertools import combinations, product

def critical_pairs(problem):
    targets=[]
    for b in problem["bounds"]:
        u=0;target=[]
        for v in b:
            u=max(u,v);target.append(u)
        targets.append(target)
    pairs=[]
    for s,q in combinations(range(len(targets)),2):
        different=[t for t,(a,b) in enumerate(zip(targets[s],targets[q])) if a!=b]
        if not different:
            continue
        deadline=different[0]
        first_service=next(j for j,(a,b) in enumerate(zip(problem["paths"][s],problem["paths"][q])) if a!=b)
        if deadline<=first_service:
            pairs.append((s,q,deadline))
    return targets,pairs

def solve(problem,lead,delay,limit,gamma):
    import numpy as np
    from scipy.optimize import Bounds,LinearConstraint,milp
    from scipy.sparse import csc_matrix
    targets,pairs=critical_pairs(problem)
    h=problem["h"]
    variables=[(r,j) for r in range(h-delay) for j in range(r+delay,min(h,r+lead+1))]
    if not pairs:
        return dict(feasible=True,requests=0,schedule=[],pairs=0)
    rows=[];lo=[];up=[]
    for s,q,t in pairs:
        rows.append([abs(problem["paths"][s][j]-problem["paths"][q][j])
                     if r+delay<=t else 0 for r,j in variables])
        lo.append(2*gamma+1);up.append(np.inf)
    for r in range(h):
        rows.append([int(rr==r) for rr,j in variables]);lo.append(0);up.append(limit)
    for j in range(h):
        rows.append([int(jj==j) for r,jj in variables]);lo.append(0);up.append(1)
    if not variables:
        return dict(feasible=False,requests=None,schedule=[],pairs=len(pairs))
    result=milp(np.ones(len(variables)),integrality=np.ones(len(variables)),
                bounds=Bounds(0,1),constraints=LinearConstraint(csc_matrix(np.array(rows,dtype=float)),lo,up),
                options=dict(time_limit=20))
    assert result.status in (0,2),result.message
    if result.status==2:
        return dict(feasible=False,requests=None,schedule=[],pairs=len(pairs))
    selected=[variables[i] for i,x in enumerate(result.x) if x>.5]
    assert all(sum(abs(problem["paths"][s][j]-problem["paths"][q][j])
                       for r,j in selected if r+delay<=t)>2*gamma for s,q,t in pairs)
    return dict(feasible=True,requests=len(selected),schedule=selected,pairs=len(pairs))

def balls_intersect(a,b,gamma):
    """Enumerate reports independently of the distance inequality."""
    for y in product(*(range(min(x,z),max(x,z)+1) for x,z in zip(a,b))):
        if sum(abs(x-v) for x,v in zip(a,y))<=gamma and sum(abs(x-v) for x,v in zip(b,y))<=gamma:
            return True
    return False

def metric_smoke():
    checks=0
    for length in (1,2,3):
        vectors=list(product(range(3),repeat=length))
        for a,b in product(vectors,repeat=2):
            for gamma in (0,1,2):
                assert balls_intersect(a,b,gamma)==(sum(abs(x-z) for x,z in zip(a,b))<=2*gamma)
                checks+=1
    return checks
