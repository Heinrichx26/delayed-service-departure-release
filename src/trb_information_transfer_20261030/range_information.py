"""Finite-calendar request optimization with range-valid release actions."""
from functools import lru_cache
from time import perf_counter
from range_release import requirement, physical, ROOT
import sys
sys.path.insert(0,str(ROOT/'src/trb_error_20261030'))
from robust_reports import SearchLimit


class RangeInformation:
    def __init__(self,paths,upper,n,cap,delay,budget=3,depth=None,seconds_limit=30):
        self.paths=tuple(tuple(c) for c in paths)
        self.upper=tuple(upper);self.n=n;self.cap=cap;self.h=len(upper)
        self.caps=(cap,)*self.h;self.delay=delay;self.budget=budget;self.depth=depth
        self.seconds_limit=seconds_limit;self.started=perf_counter();self.choice={}
        early=tuple(min(n,(j+1)*cap) for j in range(self.h))
        self.refs=tuple(physical(early,c,6) for c in self.paths)
        for name in ('members','children','value','look','none','release'):
            setattr(self,name,lru_cache(None)(getattr(self,name)))

    def members(self,past,reports):
        return tuple((s,sum(abs(c[j]-y) for j,y in reports)) for s,c in enumerate(self.paths)
                     if c[:len(past)]==past and sum(abs(c[j]-y) for j,y in reports)<=1)

    def release(self,t,past,reports,u):
        b=requirement(self.upper,self.n,self.caps,6,past,reports,1)
        assert b is not None
        return max(u,b[t])

    def actions(self,t,past,reports,pending,queried):
        yield None
        if len(queried)>=self.budget:return
        for j in range(t+self.delay,min(self.h,t+9)):
            if j not in queried and self.upper[j]>0:
                yield j

    def children(self,t,past,reports,pending,queried,action):
        combined=pending+(((t+self.delay,action),) if action is not None else ())
        due=tuple(j for d,j in combined if d==t+1)
        future=tuple((d,j) for d,j in combined if d>t+1)
        q=tuple(sorted(queried+((action,) if action is not None else ())))
        outcomes=set()
        for s,used in self.members(past,reports):
            c=self.paths[s]
            for e in range(-(1-used),2-used) if due else (0,):
                if due:
                    j=due[0];y=c[j]+e
                    if not 0<=y<=self.n:continue
                    rr=tuple(sorted(reports+((j,y),)))
                else:rr=reports
                outcomes.add((past+(c[t],),rr))
        result=[]
        for pp,rr in sorted(outcomes):
            members=self.members(pp,rr)
            assert members
            ref={self.refs[s][t] for s,_ in members}
            assert len(ref)==1
            result.append((pp,rr,future,q,ref.pop()))
        return tuple(result)

    def check(self):
        if perf_counter()-self.started>self.seconds_limit or self.value.cache_info().misses>150000:
            raise SearchLimit('Information optimization allowance reached')

    def terminal_cost(self,t,past,reports):
        return max(sum(self.n-self.refs[s][j] for j in range(t,self.h))
                   for s,_ in self.members(past,reports))

    def none(self,t,past,reports,pending,queried,u):
        if t==self.h:return 0
        z=self.release(t,past,reports,u)
        if z==self.n:return self.terminal_cost(t,past,reports)
        return max(z-ref+self.none(t+1,pp,rr,pn,q,z)
                   for pp,rr,pn,q,ref in self.children(t,past,reports,pending,queried,None))

    def look(self,t,past,reports,pending,queried,u,depth):
        self.check()
        if t==self.h:return 0
        if depth==0:return self.none(t,past,reports,pending,queried,u)
        z=self.release(t,past,reports,u)
        if z==self.n:return self.terminal_cost(t,past,reports)
        return min(max(z-ref+self.look(t+1,pp,rr,pn,q,z,depth-1)
                       for pp,rr,pn,q,ref in self.children(t,past,reports,pending,queried,a))
                   for a in self.actions(t,past,reports,pending,queried))

    def value(self,t,past,reports,pending,queried,u):
        self.check()
        if t==self.h:return 0
        state=(t,past,reports,pending,queried,u)
        z=self.release(t,past,reports,u)
        if z==self.n:
            self.choice[state]=None
            return self.terminal_cost(t,past,reports)
        actions=list(self.actions(t,past,reports,pending,queried))
        if self.depth is not None:
            def score(a):
                return (max(z-ref+self.look(t+1,pp,rr,pn,q,z,self.depth-1)
                            for pp,rr,pn,q,ref in self.children(t,past,reports,pending,queried,a)),
                        a is not None,a if a is not None else -1)
            actions=[min(actions,key=score)]
        best=None;selected=None
        for action in actions:
            worst=0
            for pp,rr,pn,q,ref in self.children(t,past,reports,pending,queried,action):
                worst=max(worst,z-ref+self.value(t+1,pp,rr,pn,q,z))
                if best is not None and worst>=best:break
            if best is None or worst<best:
                best=worst;selected=action
        self.choice[state]=selected
        return best

    def solve(self):
        result=self.value(0,(),(),(),(),0)
        return dict(value=result,seconds=perf_counter()-self.started,states=self.value.cache_info().misses)

    def select(self,t,past,reports,pending,queried,u):
        if self.members(past,reports):
            self.started=perf_counter()
            self.value(t,past,reports,pending,queried,u)
            return self.choice[(t,past,reports,pending,queried,u)]
        # The same deterministic deadline rule completes the policy on unseen prefixes.
        for j in range(t+6+self.delay,min(self.h,t+9)):
            if j not in queried and self.upper[j]>0 and len(queried)<self.budget:
                return j
        return None
