"""Direct scenario MIP with generated endogenous nonanticipativity constraints.

The formulation retains aircraft-release variables. It does not use backward
release bounds or the optimized request recursion. Complete report vectors
represent all admissible shared-error histories, including adaptive queries.
"""
from itertools import combinations
from math import ceil
from time import perf_counter
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


class ScenarioMIP:
    def __init__(self, problem, lead, delay, limit, gamma, total_requests=None,
                 seconds_limit=30, max_iterations=250):
        self.p=problem;self.lead=lead;self.delay=delay;self.limit=limit
        self.gamma=gamma;self.total_requests=total_requests
        self.seconds_limit=seconds_limit;self.max_iterations=max_iterations
        self.h=problem['h'];self.n=problem['n']
        self.coords=tuple(j for j in range(delay,self.h)
                          if len({a[j] for a in problem['paths']})>1)
        self.actions=tuple((t,j) for t in range(self.h)
                           for j in self.coords if t+delay<=j<=t+lead)
        self.by_time={t:tuple(k for k,(r,j) in enumerate(self.actions) if r==t)
                      for t in range(self.h)}
        self.worlds=[]
        def residuals(k,budget,prefix):
            if k==len(self.coords):
                yield prefix;return
            for v in range(-budget,budget+1):
                yield from residuals(k+1,budget-abs(v),prefix+(v,))
        for s,path in enumerate(problem['paths']):
            for errors in residuals(0,gamma,()):
                y=list(path)
                for j,e in zip(self.coords,errors):y[j]+=e
                if all(0<=x<=self.n for x in y):self.worlds.append((s,tuple(y)))
        unique={}
        for s,y in self.worlds:unique.setdefault((problem['paths'][s],y),(s,y))
        self.worlds=list(unique.values())
        self.nw=len(self.worlds);self.block=self.h+len(self.actions)
        self.nvars=1+self.nw*self.block
        self.rows=[];self.los=[];self.ups=[];self.cuts=set()
        self._base()

    def u(self,w,t):return 1+w*self.block+t
    def x(self,w,k):return 1+w*self.block+self.h+k

    def add(self,row,lo=-np.inf,up=np.inf):
        self.rows.append({k:v for k,v in row.items() if v});self.los.append(lo);self.ups.append(up)

    def _base(self):
        for w,(s,_) in enumerate(self.worlds):
            for t in range(self.h):
                row={self.u(w,t):1}
                if t:row[self.u(w,t-1)]=-1
                self.add(row,lo=0,up=self.p['caps'][t])
                if t>=self.p['lag']:
                    self.add({self.u(w,t-self.p['lag']):1},lo=self.p['refs'][s][t])
                indices=self.by_time[t]
                if indices:self.add({self.x(w,k):1 for k in indices},up=self.limit)
            self.add({self.u(w,self.h-1):1},lo=self.n,up=self.n)
            for j in self.coords:
                self.add({self.x(w,k):1 for k,(t,jj) in enumerate(self.actions) if jj==j},up=1)
            if self.total_requests is not None:
                self.add({self.x(w,k):1 for k in range(len(self.actions))},up=self.total_requests)
            row={0:1};row.update({self.u(w,t):-1 for t in range(self.h)})
            self.add(row,lo=-sum(self.p['refs'][s]))

    def add_cut(self,w,q,t):
        key=(min(w,q),max(w,q),t)
        if key in self.cuts:return False
        self.cuts.add(key)
        yw=self.worlds[w][1];yq=self.worlds[q][1]
        distinguishing=[k for k,(r,j) in enumerate(self.actions)
                        if r+self.delay<=t and yw[j]!=yq[j]]
        def equality(i,j,scale):
            for sign in (1,-1):
                row={i:sign,j:-sign}
                for k in distinguishing:row[self.x(w,k)]=row.get(self.x(w,k),0)-scale
                self.add(row,up=0)
        equality(self.u(w,t),self.u(q,t),self.n)
        for k in self.by_time[t]:equality(self.x(w,k),self.x(q,k),1)
        return True

    def violations(self,vector):
        cuts=[]
        # Induction over decisions: histories include earlier actions. Any
        # illegal branching is detected at its earliest decision time.
        for t in range(self.h):
            groups={}
            for w,(s,y) in enumerate(self.worlds):
                sent=tuple((r,j,y[j] if r+self.delay<=t else None)
                           for k,(r,j) in enumerate(self.actions)
                           if r<t and vector[self.x(w,k)]>.5)
                history=(self.p['paths'][s][:t],sent)
                decision=(int(round(vector[self.u(w,t)])),
                          tuple(k for k in self.by_time[t] if vector[self.x(w,k)]>.5))
                if history not in groups:groups[history]=(w,decision)
                else:
                    q,other=groups[history]
                    if decision!=other:cuts.append((q,w,t))
        return cuts

    def solve(self):
        start=perf_counter();last=None
        # Integer weighting gives waiting strict priority, then minimizes
        # total requests over the complete scenario list. This removes
        # cost-free redundant request choices from the generic formulation.
        max_requests=self.nw*len(self.coords)
        scale=max_requests+1
        objective=np.zeros(self.nvars);objective[0]=scale
        for w in range(self.nw):
            for k in range(len(self.actions)):objective[self.x(w,k)]=1
        def waiting_bound(result):
            raw=getattr(result,'mip_dual_bound',None)
            if raw is None or not np.isfinite(raw):return None
            return max(0,ceil((float(raw)-max_requests)/scale-1e-6))
        lb=np.zeros(self.nvars);ub=np.ones(self.nvars)
        earliest=[];released=0
        for cap in self.p['caps']:
            released=min(self.n,released+cap);earliest.append(released)
        ub[0]=max(sum(a-b for a,b in zip(earliest,ref)) for ref in self.p['refs'])
        for w in range(self.nw):ub[1+w*self.block:1+w*self.block+self.h]=self.n
        bounds=Bounds(lb,ub)
        integrality=np.ones(self.nvars)
        for iteration in range(1,self.max_iterations+1):
            remaining=self.seconds_limit-(perf_counter()-start)
            if remaining<=0:break
            rr=[];cc=[];vv=[]
            for i,row in enumerate(self.rows):
                for k,v in row.items():rr.append(i);cc.append(k);vv.append(v)
            matrix=coo_matrix((vv,(rr,cc)),shape=(len(self.rows),self.nvars)).tocsc()
            result=milp(objective,integrality=integrality,bounds=bounds,
                        constraints=LinearConstraint(matrix,self.los,self.ups),
                        options={'time_limit':remaining,'mip_rel_gap':0.0})
            last=result
            if result.x is None:break
            violating=self.violations(result.x)
            if not violating:
                self.verify(result.x)
                return dict(resolved=result.status==0,value=int(round(result.x[0])),
                            feasible=True,dual_bound=waiting_bound(result),
                            seconds=perf_counter()-start,iterations=iteration,
                            worlds=self.nw,variables=self.nvars,constraints=len(self.rows),
                            generated_pairs=len(self.cuts))
            added=sum(self.add_cut(*cut) for cut in violating)
            assert added, 'Repeated inconsistent scenario solution'
        return dict(resolved=False,feasible=False,seconds=perf_counter()-start,
                    dual_bound=waiting_bound(last) if last is not None else None,
                    worlds=self.nw,variables=self.nvars,constraints=len(self.rows),
                    generated_pairs=len(self.cuts),iterations=iteration)

    def verify(self,vector):
        assert not self.violations(vector)
        for w,(s,y) in enumerate(self.worlds):
            u=tuple(int(round(vector[self.u(w,t)])) for t in range(self.h))
            prev=done=0;trajectory=[]
            for t,c in enumerate(self.p['paths'][s]):
                assert 0<=u[t]-prev<=self.p['caps'][t];prev=u[t]
                arrived=u[t-self.p['lag']] if t>=self.p['lag'] else 0
                done=min(arrived,done+c);trajectory.append(done)
            assert tuple(trajectory)==self.p['refs'][s]
            assert u[-1]==self.n
            selected=[self.actions[k] for k in range(len(self.actions)) if vector[self.x(w,k)]>.5]
            assert len({j for t,j in selected})==len(selected)
            assert all(sum(r==t for r,j in selected)<=self.limit for t in range(self.h))
            if self.total_requests is not None:assert len(selected)<=self.total_requests
