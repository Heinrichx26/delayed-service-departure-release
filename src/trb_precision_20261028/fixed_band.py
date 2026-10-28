"""Fixed dispatch optimization; only interval answers distinguish calendars."""
from itertools import combinations
from time import perf_counter
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack, csc_matrix
from band_design import old


def fixed_band(problem, lead, delay, budget, width, seconds=30):
    started = perf_counter()
    paths = problem['paths']
    m, h, t0, n = len(paths), problem['horizon'], problem['t0'], problem['n']
    reports = [[v//width for v in p] for p in paths]
    requests = [(r, j) for r in range(t0, h-delay)
                for j in range(r+delay, min(h, r+lead+1))
                if len({p[j] for p in reports}) > 1]
    k = len(requests)
    nv = k+m*h
    rr, cc, vv, lower, upper = [], [], [], [], []
    def constraint(terms, lo, hi):
        row = len(lower)
        for c, v in terms:
            rr.append(row); cc.append(c); vv.append(v)
        lower.append(lo); upper.append(hi)
    def idx(s, t): return k+s*h+t
    for r in range(t0, h):
        constraint([(a, 1) for a, (v, j) in enumerate(requests) if r == v], 0, budget)
    for j in range(h):
        constraint([(a, 1) for a, (r, v) in enumerate(requests) if j == v], 0, 1)
    for s in range(m):
        for t in range(h):
            constraint([(idx(s,t), 1)]+([(idx(s,t-1), -1)] if t else []),
                       0, problem['caps'][t])
    for s, q in combinations(range(m), 2):
        first = next((j for j in range(h) if paths[s][j] != paths[q][j]), h)
        for t in range(t0, min(h, first+1)):
            distinguishes = [(a, -n) for a, (r,j) in enumerate(requests)
                             if r+delay <= t and reports[s][j] != reports[q][j]]
            constraint([(idx(s,t),1), (idx(q,t),-1)]+distinguishes, -np.inf, 0)
            constraint([(idx(q,t),1), (idx(s,t),-1)]+distinguishes, -np.inf, 0)
    mat = coo_matrix((vv,(rr,cc)), shape=(len(lower),nv)).tocsc()
    lb = np.r_[np.zeros(k), np.array(problem['bounds']).reshape(-1)]
    ub = np.r_[np.ones(k), np.full(m*h,n)]
    for s in range(m):
        for t,u in enumerate(problem['prefix']): lb[idx(s,t)] = ub[idx(s,t)] = u
    first = np.r_[np.zeros(k), np.repeat(problem['weights'],h)]
    for s in range(m):
        for t in range(t0): first[idx(s,t)] = 0
    witness = np.r_[np.zeros(k),np.tile(old.base.earliest(n,problem['caps'],problem['prefix']),m)]
    assert np.all(witness >= lb) and np.all(witness <= ub)
    assert np.all(mat@witness >= np.array(lower)-1e-8)
    assert np.all(mat@witness <= np.array(upper)+1e-8)
    retries = []
    def optimize(objective, matrix, lo, hi, stage):
        ans = milp(objective, integrality=np.ones(nv), bounds=Bounds(lb,ub),
                   constraints=LinearConstraint(matrix,lo,hi),
                   options=dict(time_limit=seconds,mip_rel_gap=0))
        if ans.status == 2:
            retries.append(dict(stage=stage,status=int(ans.status),message=ans.message))
            ans = milp(objective,integrality=np.ones(nv),bounds=Bounds(lb,ub),
                       constraints=LinearConstraint(matrix,lo,hi),
                       options=dict(time_limit=seconds,mip_rel_gap=0,presolve=False))
        return ans
    ans = optimize(first,mat,lower,upper,'waiting')
    if ans.status != 0:
        return dict(optimal=False,stage='waiting',status=int(ans.status),message=ans.message)
    value = int(round(ans.fun))
    second = np.r_[np.ones(k),np.zeros(m*h)]
    mat2 = vstack([mat,csc_matrix(first.reshape(1,-1))],format='csc')
    lo, hi = np.r_[lower,value],np.r_[upper,value]
    ans2 = optimize(second,mat2,lo,hi,'requests')
    if ans2.status != 0:
        return dict(optimal=False,stage='requests',status=int(ans2.status),message=ans2.message)
    solution = np.rint(ans2.x).astype(int)
    assert np.all(mat2@solution >= lo-1e-7) and np.all(mat2@solution <= hi+1e-7)
    queries = [dict(dispatch=r,arrival=r+delay,slot=j)
               for a,(r,j) in enumerate(requests) if solution[a]]
    rows = []
    for s,path in enumerate(paths):
        release = solution[k+s*h:k+(s+1)*h].tolist()
        done,inventory,_ = old.base.physical(release,path,n,problem['lag'])
        assert done == problem['references'][s]
        rows.append(dict(s=s,queries=queries,release=release,completed=done,
                         post_release=problem['case']['delta']*sum(inventory[t0:])/n))
    const = sum(w*sum(ref[t0:]) for w,ref in zip(problem['weights'],problem['references']))
    return dict(optimal=True,weighted_waiting=value-const,
                waiting=(value-const)*problem['case']['delta']/problem['denominator']/n,
                count=len(queries),paths=rows,retries=retries,seconds=perf_counter()-started)
