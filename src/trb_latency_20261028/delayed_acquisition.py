"""Exact information acquisition with delayed answers and bounded dispatches."""
from __future__ import annotations
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/problem_driven_20261028'))
import info_design as old

class StateLimit(RuntimeError):
    pass

def delayed_targets(problem, lead, delay):
    """All available requests dispatched from t0, answers after delay slots."""
    paths, h, t0 = problem['paths'], problem['horizon'], problem['t0']
    targets = [list(problem['prefix']) for _ in paths]
    for t in range(t0, h):
        length = t if t < t0+delay else min(h, max(t, t+lead-delay+1))
        groups = defaultdict(list)
        for s, path in enumerate(paths):
            groups[tuple(path[:length])].append(s)
        for members in groups.values():
            bound = max(problem['bounds'][s][t] for s in members)
            for s in members:
                previous = targets[s][-1] if targets[s] else 0
                targets[s].append(max(previous, bound))
    for s, release in enumerate(targets):
        assert all(0 <= u <= k for u,k in zip(old.base.increments(release),problem['caps']))
        assert old.base.physical(release, paths[s], problem['n'],problem['lag'])[0] == problem['references'][s]
    return targets

class DelayedDesign:
    def __init__(self, problem, lead, delay, budget, targets=None, state_limit=150000, seconds_limit=40):
        assert delay >= 1 and budget >= 1
        self.problem, self.lead, self.delay, self.budget = problem, lead, delay, budget
        self.paths = problem['paths']
        self.target = delayed_targets(problem,lead,delay) if targets is None else targets
        self.h, self.t0 = problem['horizon'], problem['t0']
        self.weights, self.denominator = problem['weights'], problem['denominator']
        self.full = (1 << len(self.paths))-1
        altered = dict(problem,target=self.target)
        self.pairs = old.critical_pairs(altered)
        self.choice, self.values = {}, {}
        self.state_limit, self.seconds_limit = state_limit, seconds_limit
        self.started = perf_counter()
        self.part = lru_cache(None)(self.part)
        self.weight = lru_cache(None)(self.weight)
        self.value = lru_cache(None)(self.value)

    def weight(self, mask):
        return sum(w for s,w in enumerate(self.weights) if mask >> s & 1)

    def part(self, mask, slots):
        groups = defaultdict(int)
        for s,path in enumerate(self.paths):
            if mask >> s & 1:
                groups[tuple(path[j] for j in slots)] |= 1 << s
        return tuple(sorted(groups.values()))

    def canonical(self, mask, pending):
        return tuple(sorted((due,j) for due,j in pending if len(self.part(mask,(j,)))>1))

    def transition(self, t, mask, pending, request):
        combined = tuple(pending)+tuple((t+self.delay,j) for j in request)
        due = tuple(j for arrival,j in combined if arrival==t+1)
        future = tuple((arrival,j) for arrival,j in combined if arrival>t+1)
        return tuple((sub,self.canonical(sub,future)) for sub in self.part(mask,(t,)+due))

    def outstanding(self,t,mask,pending):
        return [(s,q,deadline) for s,q,deadline in self.pairs
                if deadline>=t and mask>>s&1 and mask>>q&1 and
                not any(due<=deadline and self.paths[s][j]!=self.paths[q][j] for due,j in pending)]

    def value(self,t,mask,pending):
        state = (t,mask,pending)
        if len(self.values)>self.state_limit or (len(self.values)%256==0 and perf_counter()-self.started>self.seconds_limit):
            raise StateLimit(f'states={len(self.values)},seconds={perf_counter()-self.started:.2f}')
        if t==self.h:
            self.choice[state]=();self.values[state]=0;return 0
        if len({self.target[s][t] for s in range(len(self.paths)) if mask>>s&1})>1:
            self.choice[state]=None;self.values[state]=None;return None
        required = self.outstanding(t,mask,pending)
        if not required:
            self.choice[state]=();self.values[state]=0;return 0
        if min(deadline for _,_,deadline in required)<t+self.delay:
            self.choice[state]=None;self.values[state]=None;return None
        existing={j for _,j in pending}
        candidates=[];seen=set()
        for j in range(t+self.delay,min(self.h,t+self.lead+1)):
            if j in existing:
                continue
            partition=self.part(mask,(j,))
            if len(partition)<=1 or partition in seen:
                continue
            cover=sum(1<<i for i,(s,q,d) in enumerate(required) if t+self.delay<=d and self.paths[s][j]!=self.paths[q][j])
            if cover:
                seen.add(partition);candidates.append((j,cover))
        urgent=sum(1<<i for i,(_,_,d) in enumerate(required) if d==t+self.delay)
        best,selected=None,None
        for size in range(min(self.budget,len(candidates))+1):
            for indices in combinations(range(len(candidates)),size):
                request=tuple(candidates[k][0] for k in indices)
                covered=0
                for k in indices:covered|=candidates[k][1]
                if covered&urgent!=urgent:
                    continue
                if len(self.part(mask,request))==1 and request:
                    continue
                cost=self.weight(mask)*size
                if best is not None and cost>=best:
                    continue
                valid=True
                for sub,future in self.transition(t,mask,pending,request):
                    child=self.value(t+1,sub,future)
                    if child is None:
                        valid=False;break
                    cost+=child
                    if best is not None and cost>=best:
                        valid=False;break
                if valid and (best is None or cost<best):
                    best,selected=cost,request
        self.values[state]=best;self.choice[state]=selected
        return best

    def initial_groups(self):
        return self.part(self.full,tuple(range(self.t0)))

    def solve(self):
        values=[self.value(self.t0,mask,()) for mask in self.initial_groups()]
        feasible=all(v is not None for v in values)
        total=sum(values) if feasible else None
        return dict(feasible=feasible,weighted_cost=total,
                    expected_queries=float(Fraction(total,self.denominator)) if feasible else None,
                    expected_queries_exact=str(Fraction(total,self.denominator)) if feasible else None,
                    states=len(self.values),seconds=perf_counter()-self.started)

    def replay(self):
        rows=[]
        for s,path in enumerate(self.paths):
            mask=next(m for m in self.initial_groups() if m>>s&1)
            pending=();queries=[];release=list(self.problem['prefix'])
            for t in range(self.t0,self.h):
                assert self.value(t,mask,pending) is not None
                request=self.choice[(t,mask,pending)]
                assert len(request)<=self.budget
                for j in request:
                    assert j-self.lead<=t and j>=t+self.delay
                    queries.append(dict(dispatch=t,arrival=t+self.delay,slot=j))
                values={self.target[q][t] for q in range(len(self.paths)) if mask>>q&1}
                assert len(values)==1
                release.append(next(iter(values)))
                children=self.transition(t,mask,pending,request)
                mask,pending=next((m,p) for m,p in children if m>>s&1)
            assert release==self.target[s]
            completed,inventory,_=old.base.physical(release,path,self.problem['n'],self.problem['lag'])
            assert completed==self.problem['references'][s]
            assert all(0<=u<=k for u,k in zip(old.base.increments(release),self.problem['caps']))
            rows.append(dict(s=s,queries=queries,query_count=len(queries),release=release,completed=completed,
                             post_release=self.problem['case']['delta']*sum(inventory)/self.problem['n']))
        weighted=sum(self.weights[r['s']]*r['query_count'] for r in rows)
        assert weighted==sum(self.value(self.t0,m,()) for m in self.initial_groups())
        return rows

def fixed_schedule(problem,lead,delay,budget,target,seconds=20):
    """Exact binary covering and dispatch-capacity formulation."""
    import numpy as np
    from scipy.optimize import Bounds,LinearConstraint,milp
    from scipy.sparse import csc_matrix
    h,t0,paths=problem['horizon'],problem['t0'],problem['paths']
    pairs=old.critical_pairs(dict(problem,target=target))
    if not pairs:return dict(feasible=True,optimal=True,count=0,schedule=[],pairs=0)
    variables=[]
    for t in range(t0,h-delay):
        for j in range(t+delay,min(h,t+lead+1)):
            cover=tuple(i for i,(s,q,d) in enumerate(pairs) if t+delay<=d and paths[s][j]!=paths[q][j])
            if cover:variables.append((t,j,cover))
    if not variables:return dict(feasible=False,optimal=True,pairs=len(pairs),schedule=[])
    rr=[];cc=[];vv=[]
    for col,(t,j,cover) in enumerate(variables):
        for row in cover:rr.append(row);cc.append(col);vv.append(1.)
        rr.extend([len(pairs)+t,len(pairs)+h+j]);cc.extend([col,col]);vv.extend([1.,1.])
    matrix=csc_matrix((vv,(rr,cc)),shape=(len(pairs)+2*h,len(variables)))
    lower=np.r_[np.ones(len(pairs)),np.full(2*h,-np.inf)]
    upper=np.r_[np.full(len(pairs),np.inf),np.full(h,budget),np.ones(h)]
    result=milp(np.ones(len(variables)),integrality=np.ones(len(variables)),
                bounds=Bounds(0,1),constraints=LinearConstraint(matrix,lower,upper),
                options=dict(time_limit=seconds,mip_rel_gap=0))
    if result.status==2:return dict(feasible=False,optimal=True,pairs=len(pairs),schedule=[])
    if result.status!=0:return dict(feasible=None,optimal=False,pairs=len(pairs),status=int(result.status),message=str(result.message))
    schedule=[dict(dispatch=t,arrival=t+delay,slot=j) for take,(t,j,_) in zip(result.x,variables) if take>.5]
    assert len(schedule)==round(result.fun)
    assert all(sum(q['dispatch']==t for q in schedule)<=budget for t in range(h))
    assert len({q['slot'] for q in schedule})==len(schedule)
    for s,path in enumerate(paths):
        for t in range(t0,h):
            observed=tuple(range(t))+tuple(q['slot'] for q in schedule if q['arrival']<=t)
            compatible=[q for q,p in enumerate(paths) if all(p[j]==path[j] for j in observed)]
            assert len({target[q][t] for q in compatible})==1
    return dict(feasible=True,optimal=True,count=len(schedule),schedule=schedule,pairs=len(pairs),variables=len(variables))
