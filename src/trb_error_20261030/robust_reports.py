"""Finite-calendar minimax request/release optimization with erroneous reports.

The error allowance bounds the sum of absolute count errors across issued
reports, not calendar error or elapsed controller working time. Free past
service observations remain exact. No distribution of report errors is used.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations, product
from time import perf_counter


class SearchLimit(RuntimeError):
    pass


def physical(releases, service, lag):
    """Independent cumulative takeoff simulation; releases are cumulative."""
    done = []
    previous = 0
    for t, c in enumerate(service):
        arrived = releases[t-lag] if t >= lag else 0
        previous = min(arrived, previous+c)
        done.append(previous)
    return tuple(done)


def prepare(paths, n, caps, lag=0):
    paths = tuple(tuple(map(int, p)) for p in paths)
    h = len(caps)
    assert paths and all(len(p) == h for p in paths)
    assert all(0 <= x <= n for p in paths for x in p)
    earliest, previous = [], 0
    for k in caps:
        previous = min(n, previous+k)
        earliest.append(previous)
    refs = tuple(physical(earliest, p, lag) for p in paths)
    if any(r[-1] != n for r in refs):
        raise ValueError("At least one service calendar cannot complete the group")
    bounds = []
    for ref in refs:
        b = [ref[min(t+lag, h-1)] for t in range(h)]
        for t in range(h-2, -1, -1):
            b[t] = max(b[t], b[t+1]-caps[t+1])
        bounds.append(tuple(b))
    return dict(paths=paths, n=n, caps=tuple(caps), lag=lag, h=h,
                refs=refs, bounds=tuple(bounds))


def error_vectors(length, allowance):
    """All integer response residuals with total absolute error <= allowance."""
    if length == 0:
        yield ()
        return
    for first in range(-allowance, allowance+1):
        for tail in error_vectors(length-1, allowance-abs(first)):
            yield (first,) + tail


class RobustReports:
    def __init__(self, problem, lead, delay, dispatch_limit, error_allowance,
                 price=0, state_limit=120000, seconds_limit=25, mode="exact",
                 independent_errors=False):
        assert delay >= 1 and dispatch_limit >= 0 and error_allowance >= 0
        assert isinstance(price, int) and price >= 0
        self.p = problem
        self.paths = problem["paths"]
        self.h, self.n = problem["h"], problem["n"]
        self.lead, self.delay, self.limit = lead, delay, dispatch_limit
        self.gamma, self.price, self.mode = error_allowance, price, mode
        self.independent_errors = independent_errors
        self.state_limit, self.seconds_limit = state_limit, seconds_limit
        self.started = perf_counter()
        self.choice = {}
        self.transitions = lru_cache(None)(self.transitions)
        self.value = lru_cache(None)(self.value)
        self.lower = lru_cache(None)(self.lower)
        self.stats = dict(branches=0, bound_prunes=0)

    def initial(self):
        return (0, (0,)*len(self.paths), (), 0, 0)

    def members(self, spent):
        return tuple(i for i, x in enumerate(spent) if x >= 0)

    def release(self, t, spent, previous):
        z = max(previous, max(self.p["bounds"][s][t]
                              for s in self.members(spent)))
        assert 0 <= z-previous <= self.p["caps"][t]
        return z

    def batches(self, t, spent, pending, queried):
        # A slot is queried at most once. Exact past service is received free.
        candidates = tuple(j for j in range(t+self.delay, min(self.h,t+self.lead+1))
                           if not (queried >> j & 1) and
                           len({self.paths[s][j] for s in self.members(spent)})>1)
        yield ()
        for k in range(1,min(self.limit,len(candidates))+1):
            yield from combinations(candidates,k)

    def transitions(self, t, spent, pending, queried, batch):
        pending = tuple(sorted(pending+tuple((t+self.delay,j) for j in batch)))
        queried |= sum(1 << j for j in batch)
        queried &= ~((1 << min(self.h,t+1+self.delay))-1)
        due = tuple(j for d,j in pending if d == t+1)
        future = tuple((d,j) for d,j in pending if d > t+1)
        outcomes = set()
        for s in self.members(spent):
            residuals = (product(range(-self.gamma,self.gamma+1),repeat=len(due))
                         if self.independent_errors else
                         error_vectors(len(due), self.gamma-spent[s]))
            for residual in residuals:
                y = tuple(self.paths[s][j]+e for j,e in zip(due,residual))
                if all(0 <= a <= self.n for a in y):
                    outcomes.add((self.paths[s][t], y))
        result = []
        for observed, y in sorted(outcomes):
            next_spent = []
            for s, e in enumerate(spent):
                if e < 0 or self.paths[s][t] != observed:
                    next_spent.append(-1)
                    continue
                residuals=tuple(abs(a-self.paths[s][j]) for a,j in zip(y,due))
                new = (max(residuals,default=0) if self.independent_errors else
                       e+sum(residuals))
                if self.independent_errors:
                    next_spent.append(0 if new<=self.gamma else -1)
                    continue
                next_spent.append(new if new <= self.gamma else -1)
            next_spent = tuple(next_spent)
            indices = self.members(next_spent)
            assert indices
            reference = {self.p["refs"][s][t] for s in indices}
            assert len(reference) == 1
            result.append((observed, y, next_spent, future, queried, reference.pop()))
        return tuple(result)

    def lower(self, t, spent, previous):
        # Perfect future service information supplies an admissible lower bound.
        costs = []
        for s in self.members(spent):
            z, total = previous, 0
            for k in range(t,self.h):
                z = max(z,self.p["bounds"][s][k])
                total += z-self.p["refs"][s][k]
            costs.append(total)
        return max(costs, default=0)

    def value(self, t, spent, pending, queried, previous):
        if t == self.h:
            return 0
        if self.value.cache_info().misses > self.state_limit:
            raise SearchLimit("state allowance reached")
        if perf_counter()-self.started > self.seconds_limit:
            raise SearchLimit("time allowance reached")
        state = (t,spent,pending,queried,previous)
        z = self.release(t,spent,previous)
        if z == self.n:
            # Requests no longer affect any release; wait for physical service.
            answer = max(sum(self.n-self.p["refs"][s][k] for k in range(t,self.h))
                         for s in self.members(spent))
            self.choice[state] = ()
            return answer
        actions = list(self.batches(t,spent,pending,queried))
        if self.mode == "none":
            actions = [()]
        if self.mode == "greedy":
            # Choose a batch by perfect-information continuation after it returns.
            # Re-evaluate this same rule at each later information state.
            def score(batch):
                children = self.transitions(t,spent,pending,queried,batch)
                return (max(z-r+self.lower(t+1,e,z) for _,_,e,_,_,r in children)
                        + self.price*len(batch),len(batch),batch)
            actions = [min(actions,key=score)]
        best, selected = None, None
        for batch in actions:
            children = self.transitions(t,spent,pending,queried,batch)
            lower = self.price*len(batch)+max(z-r+self.lower(t+1,e,z)
                                              for _,_,e,_,_,r in children)
            if best is not None and lower >= best:
                self.stats["bound_prunes"] += 1
                continue
            worst = 0
            for _,_,e,pend,q,r in children:
                self.stats["branches"] += 1
                cost = z-r+self.value(t+1,e,pend,q,z)
                worst = max(worst,cost)
                if best is not None and self.price*len(batch)+worst >= best:
                    break
            candidate = self.price*len(batch)+worst
            if best is None or candidate < best:
                best, selected = candidate, batch
        assert best is not None
        self.choice[state] = selected
        return best

    def solve(self):
        value = self.value(*self.initial())
        return dict(value=value,states=self.value.cache_info().misses,
                    seconds=perf_counter()-self.started,**self.stats)

    def verify(self):
        """Evaluate each calendar and every allowed response in the policy graph."""
        self.solve()
        checks = 0
        nodes = 0
        nominal = []
        for actual in range(len(self.paths)):
            path = self.paths[actual]
            @lru_cache(None)
            def visit(t,spent,pending,queried,previous,actual_error):
                nonlocal checks,nodes
                nodes += 1
                if t == self.h:
                    return (0,0,0,0)
                state=(t,spent,pending,queried,previous)
                self.value(*state)
                batch=self.choice[state]
                z=self.release(t,spent,previous)
                assert spent[actual] == actual_error
                assert 0 <= z-previous <= self.p["caps"][t]
                assert z >= self.p["bounds"][actual][t]
                checks += 1
                children=self.transitions(t,spent,pending,queried,batch)
                vals=[]
                for observed,y,e,pend,q,ref in children:
                    if e[actual] < 0:
                        continue
                    assert observed == path[t]
                    child=visit(t+1,e,pend,q,z,e[actual])
                    vals.append((z-ref+child[0],len(batch)+child[1],
                                 z-ref+self.price*len(batch)+child[2],child[3]))
                assert vals
                return (max(v[0] for v in vals),max(v[1] for v in vals),
                        max(v[2] for v in vals),0)
            worst=visit(*self.initial(),0)
            state=self.initial();release=[];requests=0
            for t in range(self.h):
                _,e,pend,q,u=state
                self.value(*state)
                batch=self.choice[state]; requests+=len(batch)
                z=self.release(t,e,u); release.append(z)
                child=next(x for x in self.transitions(t,e,pend,q,batch)
                           if x[2][actual] == 0 and x[0] == path[t])
                state=(t+1,child[2],child[3],child[4],z)
            assert physical(release,path,self.p["lag"]) == self.p["refs"][actual]
            nominal.append(dict(s=actual,waiting=sum(a-b for a,b in zip(release,self.p["refs"][actual])),
                                requests=requests,worst_waiting=worst[0],
                                worst_requests=worst[1],worst_value=worst[2]))
        assert max(x["worst_value"] for x in nominal) == self.value(*self.initial())
        return dict(passed=True,physical_checks=checks,policy_nodes=nodes,paths=nominal)
