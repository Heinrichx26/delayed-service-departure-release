"""Joint minimum-waiting and minimum-request departure decisions.

All values are integer weighted aircraft-slots and weighted request counts.
The physical simulator in the earlier study is used only for verification.
"""
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
from .delayed_acquisition import DelayedDesign, StateLimit, old


class JointDesign(DelayedDesign):
    def __init__(self, problem, lead, delay, budget, state_limit=250000, seconds_limit=60):
        super().__init__(problem, lead, delay, budget, targets=problem['target'],
                         state_limit=state_limit, seconds_limit=seconds_limit)
        self.bound = lru_cache(None)(self.bound)
        self.ref = lru_cache(None)(self.ref)
        self.candidates = lru_cache(None)(self.candidates)
        self.suffix = lru_cache(None)(self.suffix)

    def bound(self, mask, t):
        return max(self.problem['bounds'][s][t]
                   for s in range(len(self.paths)) if mask >> s & 1)

    def ref(self, mask, t):
        return sum(self.weights[s]*self.problem['references'][s][t]
                   for s in range(len(self.paths)) if mask >> s & 1)

    def candidates(self, t, mask, pending):
        existing = {j for _,j in pending}
        seen = set()
        result = []
        for j in range(t+self.delay, min(self.h, t+self.lead+1)):
            if j in existing:
                continue
            partition = self.part(mask, (j,))
            if len(partition) > 1 and partition not in seen:
                seen.add(partition)
                result.append(j)
        return tuple(result)

    def suffix(self, t, mask, previous):
        """Exact zero-request value when future release bounds all agree."""
        total = 0
        for k in range(t,self.h):
            bs = {self.problem['bounds'][s][k] for s in range(len(self.paths)) if mask >> s & 1}
            if len(bs)>1 and previous<self.problem['n']:
                return None
            previous = max(previous, max(bs))
            total += self.weight(mask)*previous-self.ref(mask,k)
        return total,0

    def value(self, t, mask, pending, previous):
        state = (t,mask,pending,previous)
        if len(self.values)>self.state_limit or (len(self.values)%256==0 and perf_counter()-self.started>self.seconds_limit):
            raise StateLimit(f'states={len(self.values)},seconds={perf_counter()-self.started:.2f}')
        if t==self.h:
            self.values[state]=(0,0); self.choice[state]=()
            return 0,0
        shortcut = self.suffix(t,mask,previous)
        if shortcut is not None:
            self.values[state]=shortcut; self.choice[state]=()
            return shortcut
        release = max(previous,self.bound(mask,t))
        assert release-previous <= self.problem['caps'][t]
        stage = self.weight(mask)*release-self.ref(mask,t)
        candidates = self.candidates(t,mask,pending)
        best = None; selected = None
        for size in range(min(self.budget,len(candidates))+1):
            for request in combinations(candidates,size):
                if request and len(self.part(mask,request))==1:
                    continue
                cost = [stage,self.weight(mask)*size]
                for sub,future in self.transition(t,mask,pending,request):
                    child = self.value(t+1,sub,future,release)
                    cost[0]+=child[0]; cost[1]+=child[1]
                    if best is not None and (cost[0]>best[0] or (cost[0]==best[0] and cost[1]>=best[1])):
                        break
                else:
                    pair=tuple(cost)
                    if best is None or pair<best:
                        best=pair; selected=request
        assert best is not None
        self.values[state]=best; self.choice[state]=selected
        return best

    def solve(self):
        previous=self.problem['prefix'][-1] if self.problem['prefix'] else 0
        values=[self.value(self.t0,m,(),previous) for m in self.initial_groups()]
        total=tuple(sum(v[k] for v in values) for k in (0,1))
        return dict(optimal=True,weighted_waiting=total[0],weighted_queries=total[1],
                    waiting=total[0]*self.problem['case']['delta']/self.denominator/self.problem['n'],
                    expected_queries=total[1]/self.denominator,states=len(self.values),
                    seconds=perf_counter()-self.started)

    def replay(self):
        rows=[]
        for s,path in enumerate(self.paths):
            mask=next(m for m in self.initial_groups() if m>>s&1)
            pending=(); queries=[]; release=list(self.problem['prefix'])
            previous=release[-1] if release else 0
            for t in range(self.t0,self.h):
                self.value(t,mask,pending,previous)
                request=self.choice[(t,mask,pending,previous)]
                current=max(previous,self.bound(mask,t))
                release.append(current)
                for j in request:
                    queries.append(dict(dispatch=t,arrival=t+self.delay,slot=j))
                mask,pending=next((m,p) for m,p in self.transition(t,mask,pending,request) if m>>s&1)
                previous=current
            done,inventory,_=old.base.physical(release,path,self.problem['n'],self.problem['lag'])
            assert done==self.problem['references'][s]
            assert all(0<=u<=k for u,k in zip(old.base.increments(release),self.problem['caps']))
            rows.append(dict(s=s,queries=queries,release=release,completed=done,
                             post_release=self.problem['case']['delta']*sum(inventory[self.t0:])/self.problem['n']))
        answer=self.solve()
        assert sum(self.weights[r['s']]*len(r['queries']) for r in rows)==answer['weighted_queries']
        assert sum(self.weights[r['s']]*sum(r['release'][t]-r['completed'][t] for t in range(self.t0,self.h)) for r in rows)==answer['weighted_waiting']
        return rows
